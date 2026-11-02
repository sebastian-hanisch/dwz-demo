"""Offenes Feld mit DWZ-Auswertung: Saisons aus Turnieren, Erst-DWZ der Neuen, Abgang und Zugang, Lernen der Jugend, Anhebung, Referenzspieler (Anker).

Aufbau: `n` Spieler mit wahren Stärken N(1500; 200) und der Startwertung 1500 (Index > 10) sowie ein Platz `n` für den Anker (konstante Stärke 1500). Eine Saison besteht aus zwei
Turnieren zu 9 Runden, an denen alle teilnehmen („alle“), oder aus zwei Turnieren zu 9 Runden nach Aktivität der Spieler und einem Mannschaftsturnier zu 7 Runden („Aktivität“,
WO 4.2.4.3). Jedes Turnier wird nach der WO ausgewertet: Gegner-DWZ vor dem Turnier, Folge-DWZ `R0 + K(W − We)` für Gewertete (K aus Index, Alter und Wertung), Erst-DWZ für
Ungewertete über Zusatzstufen, Restpartien und Gesamtiteration (3.4, Anhang 1.1), Sonderwertung 300 (3.6.3). Nach einer Einlaufzeit gehen je Saison Spieler, und Neue kommen
ungewertet. Zufall nur über SplitMix64; der Verbrauch je Lauf hat eine feste Reihenfolge und Größe, die nicht von den Schaltern der Regeln abhängt (gepaarte Vergleiche teilen ihre
Zufallszahlen)."""

from dataclasses import dataclass

import numpy as np

import dwz_constants as C
import dwz_rules as W
import turnier_generator as T

_SALT = 0xD2E6
TOURNAMENTS, ROUNDS = 3, 9           # feste Größe der Zufallszahlen je Saison (die Terminpläne nutzen davon so viel, wie sie brauchen)
SCHEDULES = {C.SCHED[0]: ((9, "all"), (9, "all")), C.SCHED[1]: ((9, "act"), (9, "act"), (7, "team"))}


@dataclass(frozen=True)
class Model:
    n: int = C.DEFAULT_N
    seasons: int = C.DEFAULT_SEASONS               # gemessene Saisons nach der Einlaufzeit
    burn: int = C.BURN_IN
    churn: float = C.DEFAULT_CHURN
    leave: str = C.DEFAULT_LEAVE
    gap: float = C.DEFAULT_GAP                     # Abstand der Neuen unter ihrer Spitzenstärke (Punkte); mit gap > 0 sind Neue Jugendliche, die aufholen
    rate: float = C.DEFAULT_RATE                   # Anteil des Abstands, den sie je Saison aufholen
    trend: float = C.DEFAULT_TREND                 # Änderung der Stärke aller Spieler je Saison (Punkte); Neue kommen auf dem Niveau des Feldes
    kmode: str = C.KMODE[0]
    kfix: float = C.DEFAULT_KFIX
    use_a: bool = True
    use_b: bool = True
    use_r: bool = True
    use_floor: bool = True
    special: bool = True
    sched: str = C.SCHED[0]
    ages: bool = False
    uplift_at: object = None                       # Saison (nach der Einlaufzeit), zu deren Beginn die Anhebung 2026 gilt, oder None
    anchor_join: object = None                     # Saison (nach der Einlaufzeit), zu deren Beginn der Anker beitritt, oder None
    # Schalter nur für Experimente („Was keinen Effekt hat“):
    table: bool = False
    rounding: str = "up"
    draw_rate: float = C.DRAW_RATE

    @property
    def n_out(self) -> int:
        return int(round(self.churn * self.n))

    @property
    def total(self) -> int:
        return self.burn + self.seasons

    @property
    def slots(self) -> int:
        return self.n + 1

    @property
    def fixed_k(self):
        return None if self.kmode == C.KMODE[0] else self.kfix

    @property
    def first_r(self) -> float:
        return W.FIRST_R if self.use_r else 0.0

    @property
    def floor(self):
        return W.FLOOR if self.use_floor else None


SEASON_FIELDS = ("mean_rating", "mean_true", "bias", "slope", "fit", "n_rated", "anchor", "total_end", "total_after", "entry", "entry_err", "entry_n", "leave", "leave_err",
                 "leave_n", "play_a", "play_b", "play_special", "play_rest", "floor_lift", "rounding", "uplift_gain", "anchor_in", "special_cases", "iter_step", "sum_prev")


@dataclass(frozen=True)
class Run:
    """Ein Lauf: je Saison (Einlaufzeit eingeschlossen) die Kennzahlen; Wertungen in DWZ-Punkten."""
    model: Model
    seed: int
    series: dict            # Feldname -> (total,) Array
    direct: tuple           # (bias, Steigung, fit1500) unmittelbar nach der Anhebung, sonst NaN
    final: dict             # Wertung, wahre Stärke, Alter, Index, gewertet je Platz

    def __getattr__(self, name):
        try:
            return self.__dict__["series"][name]
        except KeyError:
            raise AttributeError(name) from None


def normals(rng: T.SplitMix64, size: int) -> np.ndarray:
    """Standardnormalverteilte Zahlen per Box-Muller aus der Gleichverteilten des Generators."""
    u1 = 1.0 - rng.uniform(size)
    u2 = rng.uniform(size)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def draws(seed: int, total: int, slots: int) -> dict:
    """Alle Zufallszahlen eines Laufs in fester Reihenfolge; Index 0 der ersten Achse gehört den Gründern."""
    rng = T.SplitMix64(int(seed) ^ _SALT)
    t1, h = total + 1, slots // 2 + 1
    return {"z": normals(rng, t1 * slots).reshape(t1, slots), "age": rng.uniform(t1 * slots * 2).reshape(t1, slots, 2),
            "act": rng.uniform(t1 * slots * 3).reshape(t1, slots, 3), "team": rng.uniform(t1 * slots).reshape(t1, slots),
            "pair": rng.uniform(total * TOURNAMENTS * ROUNDS * slots).reshape(total, TOURNAMENTS, ROUNDS, slots),
            "win": rng.uniform(total * TOURNAMENTS * ROUNDS * h).reshape(total, TOURNAMENTS, ROUNDS, h),
            "part": rng.uniform(total * TOURNAMENTS * slots).reshape(total, TOURNAMENTS, slots),
            "leave": rng.uniform(total * slots).reshape(total, slots)}


def _median3(u) -> np.ndarray:
    """Beta(2, 2) als Median dreier Gleichverteilter (letzte Achse)."""
    return np.median(u, axis=-1)


class _State:
    def __init__(self, m: Model, dr: dict):
        n, p = m.n, m.slots
        self.m = m
        self.true = np.full(p, C.BASE)
        self.true[:n] = C.BASE + C.POP_SD * dr["z"][0, :n]
        self.pot = self.true.copy()
        self.R = np.full(p, C.BASE)
        self.rated = np.ones(p, dtype=bool)
        self.idx = np.full(p, 11)
        self.age = np.full(p, 35)
        if m.ages:
            young = dr["age"][0, :, 0] < 0.25
            self.age = np.where(young, 10 + (dr["age"][0, :, 1] * 16).astype(int), 26 + (dr["age"][0, :, 1] * 45).astype(int))
        self.youth = np.zeros(p, dtype=bool)
        self.act = 0.15 + 0.7 * _median3(dr["act"][0])
        self.team = dr["team"][0] < 0.5
        self.active = np.zeros(p, dtype=bool)
        self.active[:n] = True
        self.rest = [[] for _ in range(p)]

    def new_player(self, k: int, s: int, dr: dict, level: float):
        m = self.m
        pot = level + C.POP_SD * dr["z"][s + 1, k]
        self.pot[k] = pot
        self.true[k] = pot - m.gap
        young = m.gap > 0
        u = dr["age"][s + 1, k]
        self.age[k] = (8 + int(u[1] * 10)) if (young and m.ages) else ((18 + int(u[1] * 43)) if m.ages else (14 if young else 30))
        self.youth[k] = young
        self.idx[k] = 0
        self.rated[k] = False
        self.R[k] = 0.0
        self.rest[k] = []
        self.act[k] = 0.15 + 0.7 * float(_median3(dr["act"][s + 1, k]))
        self.team[k] = dr["team"][s + 1, k] < 0.5


def _games(st: _State, dr: dict, s: int, t: int, rounds: int, part: np.ndarray):
    """Partien eines Turniers: (Spieler, Gegner, Punkte) aus beiden Sichten. Paarung durch Sortieren zufälliger Schlüssel unter den Teilnehmern; bei ungerader Zahl setzt einer aus."""
    m = st.m
    cnt = int(part.sum())
    half = cnt // 2
    owner, opp, score = [], [], []
    for r in range(rounds):
        keys = np.where(part, dr["pair"][s, t, r], 2.0)
        order = np.argsort(keys, kind="stable")
        a, b = order[0:2 * half:2], order[1:2 * half:2]
        p = W.win_prob(st.true[a] - st.true[b])
        d = m.draw_rate * 4.0 * p * (1.0 - p)
        pw = p - 0.5 * d
        u = dr["win"][s, t, r, :half]
        w = np.where(u < pw, 1.0, np.where(u < pw + d, 0.5, 0.0))
        owner += [a, b]
        opp += [b, a]
        score += [w, 1.0 - w]
    if not owner:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(owner), np.concatenate(opp), np.concatenate(score)


def _first_ratings(st: _State, P, O, S, R0, now: float):
    """Erst-DWZ der Ungewerteten nach WO 3.4: Zusatzstufen, Restpartien, Gesamtiteration. Rückgabe (R1 oder NaN, Mitglieder der Auswahl II, Zusatzstufe 1, letzter Schritt)."""
    m, n_all = st.m, len(R0)
    rated, unr = st.rated, ~st.rated
    for u in range(n_all):
        if st.rest[u]:
            st.rest[u] = [g for g in st.rest[u] if now - g[2] <= W.REST_YEARS + 1e-9]
    rn = np.array([len(x) for x in st.rest], dtype=float)
    rw = np.array([sum(g[1] for g in x) for x in st.rest], dtype=float)
    mp = W.MIN_POINTS - 1e-9
    m1 = unr[P] & rated[O]
    n1 = np.bincount(P[m1], minlength=n_all) + rn
    w1 = np.bincount(P[m1], S[m1], minlength=n_all) + rw
    s1 = unr & (n1 >= W.MIN_GAMES) & (w1 >= mp)
    m2 = unr[P] & ~s1[P] & (rated[O] | s1[O])
    n2 = np.bincount(P[m2], minlength=n_all) + rn
    w2 = np.bincount(P[m2], S[m2], minlength=n_all) + rw
    s2 = unr & ~s1 & (n2 >= W.MIN_GAMES) & (w2 >= mp)
    member = s1 | s2
    r1 = np.full(n_all, np.nan)
    step = 0.0
    if member.any():
        mm = member[P] & (rated[O] | member[O])
        rows_p, rows_o, rows_s = [P[mm]], [np.where(member[O[mm]], O[mm], -1)], [S[mm]]
        rows_f = [np.where(rated[O[mm]], R0[O[mm]], 0.0)]
        for u in np.nonzero(member)[0]:
            for ropp, sc, _t in st.rest[u]:
                rows_p.append(np.array([u]))
                rows_o.append(np.array([-1]))
                rows_s.append(np.array([sc]))
                rows_f.append(np.array([ropp]))
        x, step = W.total_iteration(np.concatenate(rows_p), np.concatenate(rows_o), np.concatenate(rows_f), np.concatenate(rows_s), member, n_all, s1)
        r1[member] = np.floor(W.first_rating_correction(x, m.first_r)[member] + 0.5)
    return r1, member, s1, step


def leavers(m: Model, rated, rating, key) -> np.ndarray:
    """Plätze der `n_out` Feldspieler, die nach der Saison gehen: zufällig (nach den Schlüsseln `key`) oder die Schwächsten nach Wertung, Ungewertete zuerst, Gleichstand nach `key`."""
    n = m.n
    if m.leave == C.LEAVE[0]:
        order = np.argsort(key[:n], kind="stable")
    elif m.leave == C.LEAVE[1]:
        order = np.lexsort((key[:n], np.where(rated[:n], rating[:n], -np.inf)))
    else:
        raise ValueError(f"unbekannte Abgangsregel: {m.leave}")
    return order[:m.n_out]


def _tournament(st: _State, dr: dict, s: int, t: int, rounds: int, part: np.ndarray, now: float, acc: dict):
    m = st.m
    P, O, S = _games(st, dr, s, t, rounds, part)
    n_all = len(st.R)
    R0 = st.R.copy()
    was_rated = st.rated.copy()
    r1, member, s1, step = _first_ratings(st, P, O, S, R0, now)
    acc["iter_step"] = max(acc["iter_step"], step)
    got1 = ~np.isnan(r1)
    em = was_rated[P] & (was_rated[O] | got1[O])
    Pe, Oe, Se = P[em], O[em], S[em]
    base_opp = np.where(was_rated[Oe], R0[Oe], r1[Oe])
    ng = np.bincount(Pe, minlength=n_all)
    Wt = np.bincount(Pe, Se, minlength=n_all)
    we_norm = np.bincount(Pe, W.win_prob(R0[Pe] - base_opp, m.table), minlength=n_all)
    if m.special:
        cand = was_rated & (ng >= W.PERF_MIN_GAMES) & (Wt > 0)
        full = cand & (Wt >= ng - 1e-9)
        rh = np.floor(W.performance(Pe, base_opp, Se, n_all, full) + 0.5)
        ex = cand & (rh > R0 + W.SPECIAL_MARGIN)
        acc["special_cases"] += int(ex.sum())
        opp_eval = np.where(was_rated[Oe] & ex[Oe], rh[Oe], base_opp)
        we = np.bincount(Pe, W.win_prob(R0[Pe] - opp_eval, m.table), minlength=n_all)
    else:
        we = we_norm
    ev = was_rated & (ng >= 1)
    win = Wt >= we - 1e-12
    k, k0, a, b = W.k_factor(R0[ev], st.age[ev], st.idx[ev], win[ev], m.use_a, m.use_b, m.fixed_k)
    d = Wt[ev] - we[ev]
    rn_final, raw, rounded = W.update(R0[ev], k, Wt[ev], we[ev], m.floor, m.rounding)
    total = float((k * d).sum())
    part_a = float((a * d).sum())
    part_b = float((k0 * (b - 1.0) * d).sum())
    part_special = float((k * (we_norm[ev] - we[ev])).sum())
    acc["play_a"] += part_a
    acc["play_b"] += part_b
    acc["play_special"] += part_special
    acc["play_rest"] += total - part_a - part_b - part_special
    acc["rounding"] += float((rounded - raw).sum())
    acc["floor_lift"] += float((rn_final - rounded).sum())
    st.R[ev] = rn_final
    st.idx[ev] += 1
    new = got1
    acc["entry"] += float(r1[new].sum())
    acc["entry_err"] += float((r1[new] - st.true[new]).sum())
    acc["entry_n"] += int(new.sum())
    st.R[new] = r1[new]
    st.rated[new] = True
    st.idx[new] = 1
    # Restpartien der Nicht-Mitglieder (gegen Auswahl I) sammeln, die der Mitglieder löschen (WO 3.4.2)
    nm = (~was_rated[P]) & (~member[P]) & (was_rated[O] | s1[O])
    for p_, o_, s_ in zip(P[nm], O[nm], S[nm]):
        st.rest[p_].append((float(R0[o_]) if was_rated[o_] else float(r1[o_]), float(s_), now))
    for u in np.nonzero(member)[0]:
        st.rest[u] = []


def _season_stats(st: _State) -> tuple:
    """(Wertungsmittel, Mittel der wahren Stärke, Mittel von Wertung − Stärke, Steigung, angepasste Wertung bei Stärke 1500, Zahl der Gewerteten) der Feldspieler."""
    n = st.m.n
    mask = st.rated[:n]
    if mask.sum() < 10:
        return (np.nan,) * 5 + (float(mask.sum()),)
    r, t = st.R[:n][mask], st.true[:n][mask]
    tc = t - t.mean()
    var = float((tc ** 2).sum())
    slope = float((tc * (r - r.mean())).sum() / var) if var > 0 else np.nan
    fit = float(r.mean() + slope * (C.BASE - t.mean()))
    return float(r.mean()), float(t.mean()), float((r - t).mean()), slope, fit, float(mask.sum())


def simulate(model: Model, seed: int) -> Run:
    m = model
    if m.n < 4 or m.seasons < 1:
        raise ValueError("n und seasons müssen mindestens 4 und 1 sein")
    if m.n_out > m.n - 2:
        raise ValueError("Zu viele Abgänge")
    if m.anchor_join is not None and not 0 <= m.anchor_join < m.seasons:
        raise ValueError("Der Anker muss innerhalb der Saisons beitreten")
    if m.uplift_at is not None and not 0 <= m.uplift_at < m.seasons:
        raise ValueError("Die Anhebung muss innerhalb der Saisons liegen")
    dr = draws(seed, m.total, m.slots)
    st = _State(m, dr)
    sched = SCHEDULES[m.sched]
    series = {f: np.full(m.total, np.nan) for f in SEASON_FIELDS}
    for f in ("entry", "entry_err", "entry_n", "leave", "leave_err", "leave_n", "play_a", "play_b", "play_special", "play_rest", "floor_lift", "rounding", "uplift_gain", "anchor_in",
              "special_cases", "iter_step", "sum_prev"):
        series[f] = np.zeros(m.total)
    direct = (np.nan, np.nan, np.nan)
    level = C.BASE
    sum_prev = float(st.R[st.rated & st.active].sum())
    for s in range(m.total):
        acc = {f: 0.0 for f in ("entry", "entry_err", "entry_n", "leave", "leave_err", "leave_n", "play_a", "play_b", "play_special", "play_rest", "floor_lift", "rounding",
                                "uplift_gain", "anchor_in", "special_cases", "iter_step")}
        if m.anchor_join is not None and s == m.burn + m.anchor_join:
            k = m.n
            st.active[k] = True
            st.true[k] = st.pot[k] = C.BASE
            st.R[k], st.rated[k], st.idx[k], st.age[k], st.act[k], st.team[k] = C.BASE, True, 11, 35, 1.0, True
            acc["anchor_in"] = C.BASE
        if m.uplift_at is not None and s == m.burn + m.uplift_at:
            mask = st.rated & st.active
            new_r = W.uplift(st.R[mask])
            acc["uplift_gain"] = float((new_r - st.R[mask]).sum())
            st.R[mask] = new_r
            ms = _season_stats(st)
            direct = (ms[2], ms[3], ms[4])
        for t, (rounds, kind) in enumerate(sched):
            if kind == "all":
                part = st.active.copy()
            elif kind == "act":
                part = st.active & (dr["part"][s, t] < st.act)
            else:
                part = st.active & st.team
            _tournament(st, dr, s, t, rounds, part, s + t / len(sched), acc)
        # Kennzahlen am Ende der Saison, vor Abgang und Zugang
        rating, truth, bias, slope, fit, n_rated = _season_stats(st)
        series["mean_rating"][s], series["mean_true"][s], series["bias"][s], series["slope"][s], series["fit"][s], series["n_rated"][s] = rating, truth, bias, slope, fit, n_rated
        series["total_end"][s] = float(st.R[st.rated & st.active].sum())
        if st.active[m.n]:
            series["anchor"][s] = st.R[m.n]
        # Abgang und Zugang nach der Einlaufzeit
        n_out = m.n_out
        if n_out and s >= m.burn - 1 and s < m.total - 1:
            for k in leavers(m, st.rated, st.R, dr["leave"][s]):
                if st.rated[k]:
                    acc["leave"] += float(st.R[k])
                    acc["leave_err"] += float(st.R[k] - st.true[k])
                    acc["leave_n"] += 1
                st.new_player(int(k), s, dr, level)
        series["total_after"][s] = float(st.R[st.rated & st.active].sum())
        st.age += 1
        if m.gap > 0:
            g = st.youth[:m.n] & ((st.age[:m.n] <= W.YOUTH_AGE) if m.ages else True)
            st.true[:m.n][g] += m.rate * (st.pot[:m.n][g] - st.true[:m.n][g])
        if m.trend:
            st.true[:m.n] += m.trend
            st.pot[:m.n] += m.trend
            level += m.trend
        for f in acc:
            series[f][s] = acc[f]
        series["sum_prev"][s] = sum_prev
        sum_prev = series["total_after"][s]
    final = {"rating": st.R.copy(), "true": st.true.copy(), "age": st.age.copy(), "idx": st.idx.copy(), "rated": st.rated.copy(), "active": st.active.copy()}
    return Run(m, int(seed), series, direct, final)
