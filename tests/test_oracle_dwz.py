"""Orakel-Tests für dwz-demo (DEMO-PLAYBOOK §12): jede Kernroutine gegen einen unabhängigen Rechenweg.

- Gewinnerwartung, K, Folge-DWZ, Anhebung: reine Python-Schleifen mit `math.erf` und eigenen if/else-Regeln, die Tabelle K0 hier noch einmal aus der WO abgeschrieben;
- Turnierleistung: `scipy.optimize.brentq` je Spieler;
- Gesamtiteration der Erst-DWZ: `scipy.optimize.root` auf den gekoppelten Gleichungen We(Ri) = W;
- Turnierauswertung (`dwz_field._tournament`): ein Turnier mit fest vorgegebenen Partien gegen eine Partie-für-Partie-Rechnung in reinem Python, mit und ohne Sonderwertung;
- Bilanz der Wertungssumme: die Posten der Läufe summieren sich exakt zur Änderung der Summe (auf vielen Modellen)."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq, root

import dwz_field as F
import dwz_rules as W
from conftest import make_state, run_tournament, small

SIGMA = 200.0 * math.sqrt(2.0)
# K0 nach Index 1..10 und > 10, aus der WO Anhang 1.3 (hier unabhängig von dwz_rules.K0_TABLE abgeschrieben)
TABLE = {"jung": [60, 60, 48, 46, 44, 42, 40, 38, 36, 34, 32], "alt": [60, 60, 44, 42, 40, 38, 36, 34, 32, 30, 28], "stark": [60, 60, 41, 39, 37, 35, 33, 31, 29, 27, 25]}


def ref_p(d):
    return 0.5 * (1.0 + math.erf(d / 400.0))                      # Φ(d/σ) mit σ = 200√2


def ref_k(r0, age, idx, win, use_a=True, use_b=True):
    if age <= 25:
        group = "jung" if r0 < 2200 else "stark"
    else:
        group = "alt" if r0 < 2000 else "stark"
    k0 = TABLE[group][min(idx, 11) - 1]
    a = 0.0
    if win and use_a:
        if age <= 20 and r0 < 2000:
            a = (2000 - r0) / 30.0
        elif age >= 21 and r0 < 1600:
            a = 4.0
    b = (r0 + 800) / 2400.0 if ((not win) and r0 < 1600 and use_b) else 1.0
    return min(math.floor((k0 * b + a) * 10 + 0.5) / 10.0, 80.0)


def test_gewinnerwartung_gegen_erf():
    for d in np.linspace(-900, 900, 61):
        assert float(W.win_prob(d)) == pytest.approx(ref_p(float(d)), abs=1e-12)
    assert float(W.win_prob(0.0)) == 0.5 and float(W.win_prob(300.0) + W.win_prob(-300.0)) == pytest.approx(1.0)


def test_k_faktor_gegen_regelschleife():
    rng = np.random.default_rng(7)
    for _ in range(400):
        r0, age, idx, win = float(rng.integers(900, 2700)), int(rng.integers(8, 70)), int(rng.integers(1, 16)), bool(rng.integers(0, 2))
        use_a, use_b = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
        k, k0, a, b = W.k_factor(np.array([r0]), np.array([age]), np.array([idx]), np.array([win]), use_a, use_b)
        assert float(k[0]) == pytest.approx(ref_k(r0, age, idx, win, use_a, use_b), abs=1e-9), (r0, age, idx, win)


def test_folge_dwz_gegen_formel_und_untergrenze():
    rng = np.random.default_rng(8)
    for _ in range(200):
        r0, k, w, we = float(rng.integers(1000, 2500)), float(rng.integers(10, 80)), float(rng.integers(0, 10)) / 2, float(rng.random() * 9)
        rn, raw, rounded = W.update(np.array([r0]), np.array([k]), np.array([w]), np.array([we]))
        assert float(rn[0]) == max(math.floor(r0 + k * (w - we) + 0.5), 1100)
        assert float(raw[0]) == pytest.approx(r0 + k * (w - we)) and float(rounded[0]) <= float(rn[0])


def test_anhebung_gegen_abschnitte():
    for r in (0, 500, 1000, 1001, 1200, 1600, 2000, 2110, 2111, 2500):
        expect = r + 400 if r <= 1000 else (0.64 * r + 760 if r <= 2110 else r)
        assert float(W.uplift(np.array([float(r)]))[0]) == math.floor(expect + 0.5)


def test_turnierleistung_gegen_brentq():
    rng = np.random.default_rng(9)
    for _ in range(40):
        n = int(rng.integers(5, 10))
        opp = rng.integers(1200, 2000, n).astype(float)
        score = rng.integers(0, 3, n) / 2.0
        w = float(score.sum())
        if w == 0 or w == n:
            continue
        got = float(W.performance(np.zeros(n, dtype=int), opp, score, 1, np.array([False]))[0])
        ref = brentq(lambda x: sum(ref_p(x - o) for o in opp) - w, -3000, 8000)
        assert got == pytest.approx(max(ref, 0.0), abs=1e-4)


def test_leistung_bei_hundert_prozent_mit_fiktivem_remis():
    opp = np.array([1500.0, 1600, 1400, 1550, 1450, 1500])
    got = float(W.performance(np.zeros(6, dtype=int), opp, np.ones(6), 1, np.array([True]))[0])
    mean_opp = opp.mean()
    ref = brentq(lambda x: sum(ref_p(x - o) for o in opp) + ref_p(x - mean_opp) - 6.5, -3000, 8000)
    assert got == pytest.approx(ref, abs=1e-4)


def test_gesamtiteration_gegen_root_der_gekoppelten_gleichungen():
    rng = np.random.default_rng(10)
    for _ in range(25):
        k = int(rng.integers(2, 4))                                         # Zusatzspieler 0..k-1, dazu Etablierte als feste Gegner
        rows = []                                                           # (owner, opp_idx, fixed, score)
        for u in range(k):
            for _g in range(int(rng.integers(5, 8))):
                rows.append((u, -1, float(rng.integers(1300, 1900)), float(rng.integers(0, 3)) / 2.0))
        for u in range(k):
            for v in range(u + 1, k):
                s = float(rng.integers(0, 3)) / 2.0
                rows.append((u, v, 0.0, s))
                rows.append((v, u, 0.0, 1.0 - s))
        owner, opp_idx, fixed, score = (np.array(c) for c in zip(*rows))
        wsum = np.bincount(owner, score, minlength=k)
        ngames = np.bincount(owner, minlength=k)
        if (wsum < 0.5).any() or (wsum >= ngames - 1e-9).any():
            continue                                                       # 0 Punkte oder 100 %: Sonderfälle, eigener Test
        member = np.ones(k, dtype=bool)
        x, step = W.total_iteration(owner, opp_idx, fixed, score, member, k, tol=1e-7, max_iter=4000)

        def eqs(z):
            out = []
            for u in range(k):
                we = 0.0
                for o, oi, f, _s in rows:
                    if o != u:
                        continue
                    we += ref_p(z[u] - (z[oi] if oi >= 0 else f))
                out.append(we - wsum[u])
            return out
        sol = root(eqs, np.full(k, 1500.0), tol=1e-10)
        assert sol.success
        assert step < 1e-6 and x == pytest.approx(sol.x, abs=1e-3), (x, sol.x)


def test_erst_dwz_korrektur_gegen_formel():
    for ri in (-50, 0, 500, 1099, 1100, 1500, 1999, 2000, 2400):
        r = max(ri, 0)
        expect = r if r >= 2000 else (r + 0.08 * (2000 - r) if r >= 1100 else 1100 + 9 * 0.08 * r / 11)
        assert float(W.first_rating_correction(np.array([float(ri)]))[0]) == pytest.approx(expect)
    # stetig bei 1100 (beide Formeln 1172) und bei 2000
    assert float(W.first_rating_correction(np.array([1100.0]))[0]) == pytest.approx(1172.0)
    assert float(W.first_rating_correction(np.array([1999.999]))[0]) == pytest.approx(1999.999, abs=0.01)


# --- Turnierauswertung gegen eine Partie-für-Partie-Rechnung --------------------------------------------------------------------------------------

def ref_tournament(rating, age, idx, games, special, use_a=True, use_b=True):
    """Alle Spieler gewertet: Folge-DWZ je Spieler in reinem Python. Bei `special` ersetzt die Leistung Rh (≥ 5 Partien, mehr als 300 über R0) die Gegner-DWZ."""
    n = len(rating)
    mine = {i: [] for i in range(n)}
    for i, j, w in games:
        mine[i].append((j, w))
        mine[j].append((i, 1.0 - w))
    rh = {}
    if special:
        for i in range(n):
            gl = mine[i]
            if len(gl) < 5:
                continue
            wi = sum(w for _j, w in gl)
            if wi <= 0:
                continue
            opp = [rating[j] for j, _w in gl]
            full = wi >= len(gl) - 1e-9
            target = wi + (0.5 if full else 0.0)
            rhi = brentq(lambda x: sum(ref_p(x - o) for o in opp) + (ref_p(x - sum(opp) / len(opp)) if full else 0.0) - target, -3000, 8000)
            rhi = math.floor(max(rhi, 0.0) + 0.5)
            if rhi > rating[i] + 300:
                rh[i] = rhi
    out = []
    for i in range(n):
        gl = mine[i]
        if not gl:
            out.append(rating[i])
            continue
        wi = sum(w for _j, w in gl)
        we = sum(ref_p(rating[i] - (rh[j] if j in rh else rating[j])) for j, _w in gl)
        k = ref_k(rating[i], age[i], idx[i], wi >= we - 1e-12, use_a, use_b)
        out.append(max(math.floor(rating[i] + k * (wi - we) + 0.5), 1100))
    return out


def _random_games(rng, n, rounds):
    games = []
    for _ in range(rounds):
        perm = rng.permutation(n)
        for m in range(n // 2):
            games.append((int(perm[2 * m]), int(perm[2 * m + 1]), float(rng.integers(0, 3)) / 2.0))
    return games


@pytest.mark.parametrize("special", [False, True])
def test_turnierauswertung_gegen_schleife(special):
    rng = np.random.default_rng(11 + int(special))
    cases = 0
    for _ in range(12):
        m = small(n=10, special=special)
        rating = rng.integers(1100, 2300, 10).astype(float)
        age = rng.integers(10, 60, 10)
        idx = rng.integers(1, 15, 10)
        st, dr = make_state(m, rating, rating.copy(), age=age, idx=idx)
        games = _random_games(rng, 10, 9)
        run_tournament(st, dr, m, games, part=np.r_[np.ones(10, bool), False])
        expect = ref_tournament(list(rating), list(age), list(idx), games, special)
        assert st.R[:10] == pytest.approx(np.array(expect), abs=1e-9)
        assert (st.idx[:10] == idx + 1).all()
        cases += 1
    assert cases == 12


def test_sonderwertung_loest_aus_und_zahlt_die_summe_der_formel():
    """Ein Spieler holt 7 aus 7 gegen Gleichstarke: seine Leistung liegt weit über R0 + 300, die Gegner rechnen mit Rh; die erzeugten Punkte folgen der Formel."""
    m = small(n=8, special=True, use_a=False, use_b=False, kmode="fester K", kfix=20.0, use_floor=False)
    rating = np.full(8, 1800.0)
    st, dr = make_state(m, rating, rating.copy(), age=np.full(8, 40), idx=np.full(8, 20))
    games = [(0, j, 1.0) for j in range(1, 8)]
    acc = run_tournament(st, dr, m, games, part=np.r_[np.ones(8, bool), False])
    opp = rating[1:]
    rh = brentq(lambda x: sum(ref_p(x - o) for o in opp) + ref_p(x - opp.mean()) - 7.5, -3000, 8000)
    rh = math.floor(rh + 0.5)
    assert rh > 1800 + 300 and acc["special_cases"] == 1
    # Spieler 0 bleibt bei normaler Wertung: 20·(7 − 7·0,5); jeder Gegner verliert 20·(0 − P(1800 − Rh))
    assert st.R[0] == math.floor(1800 + 20 * (7 - 3.5) + 0.5)
    for j in range(1, 8):
        assert st.R[j] == math.floor(1800 + 20 * (0 - ref_p(1800 - rh)) + 0.5)
    created = 7 * 20 * (ref_p(0.0) - ref_p(1800 - rh))
    assert acc["play_special"] == pytest.approx(created, rel=1e-9)
    assert acc["play_special"] > 0


def test_ungleiches_k0_erzeugt_nur_die_differenz():
    """Zwei Paare, je ein Sieg bei Gleichstand (p = 0,5): Index 1 hat K0 = 60, Index > 10 hat K0 = 28 (Erwachsener unter 2000). Paar 1: +30 und −14, Summe 16 = (60 − 28)·0,5; Paar 2 ist null-summig."""
    m = small(n=4, special=False, use_a=False, use_b=False, use_r=False, use_floor=False)
    rating = np.full(4, 1800.0)
    st, dr = make_state(m, rating, rating.copy(), age=np.full(4, 40), idx=np.array([1, 12, 12, 12]))
    acc = run_tournament(st, dr, m, [(0, 1, 1.0), (2, 3, 1.0)], part=np.r_[np.ones(4, bool), False])
    assert list(st.R[:4]) == [1830.0, 1786.0, 1814.0, 1786.0]
    assert acc["play_rest"] == pytest.approx(16.0, abs=1e-9)
    assert acc["play_a"] == 0 and acc["play_b"] == 0 and acc["play_special"] == 0


# --- Erst-DWZ im Turnier ----------------------------------------------------------------------------------------------------------------------

def test_erst_dwz_eines_neulings_gegen_brentq_und_die_formel():
    m = small(n=8, special=False)
    rating = np.array([1500.0, 1600, 1400, 1700, 1500, 1550, 1450, 1500])
    rated = np.array([True] * 7 + [False])
    st, dr = make_state(m, rating, rating.copy(), age=np.full(8, 40), idx=np.full(8, 15), rated=rated)
    scores = [1.0, 0.0, 0.5, 1.0, 0.0, 0.5]                                # Neuling (Platz 7) gegen 0..5
    games = [(7, o, s) for o, s in zip(range(6), scores)]
    run_tournament(st, dr, m, games, part=np.r_[np.ones(8, bool), False])
    opp = rating[:6]
    ri = brentq(lambda x: sum(ref_p(x - o) for o in opp) - sum(scores), -3000, 8000)
    expect = math.floor((ri + 0.08 * (2000 - ri) if 1100 <= ri < 2000 else (ri if ri >= 2000 else 1100 + 9 * 0.08 * max(ri, 0) / 11)) + 0.5)
    assert st.rated[7] and st.idx[7] == 1 and st.R[7] == expect


def test_neuling_ohne_fuenf_partien_oder_ohne_halben_punkt_bleibt_ungewertet_und_sammelt_restpartien():
    m = small(n=8, special=False)
    rating = np.array([1500.0, 1600, 1400, 1700, 1500, 1550, 1450, 1500])
    rated = np.array([True] * 7 + [False])
    st, dr = make_state(m, rating, rating.copy(), age=np.full(8, 40), idx=np.full(8, 15), rated=rated)
    run_tournament(st, dr, m, [(7, o, 0.0) for o in range(6)], part=np.r_[np.ones(8, bool), False], now=1.0)
    assert not st.rated[7] and len(st.rest[7]) == 6                       # 6 Partien, 0 Punkte: keine Erst-DWZ, Restpartien gespeichert
    # im nächsten Turnier ein Remis: die Restpartien zählen mit (7 Partien, 0,5 Punkte), nach 2 Jahren verfallen sie
    run_tournament(st, dr, m, [(7, 0, 0.5)], part=np.r_[np.ones(8, bool), False], now=2.0)
    assert st.rated[7] and st.idx[7] == 1 and st.rest[7] == []
    st2, dr2 = make_state(m, rating, rating.copy(), age=np.full(8, 40), idx=np.full(8, 15), rated=rated)
    run_tournament(st2, dr2, m, [(7, o, 0.0) for o in range(6)], part=np.r_[np.ones(8, bool), False], now=1.0)
    run_tournament(st2, dr2, m, [(7, 0, 0.5)], part=np.r_[np.ones(8, bool), False], now=3.5)     # 2,5 Jahre später: verfallen, nur 1 Partie
    assert not st2.rated[7] and len(st2.rest[7]) == 1


def test_zwei_neulinge_spielen_gegeneinander_und_die_iteration_koppelt_sie():
    m = small(n=10, special=False)
    rating = np.array([1500.0, 1600, 1400, 1700, 1500, 1550, 1450, 1500, 0, 0])
    rated = np.array([True] * 8 + [False, False])
    st, dr = make_state(m, rating, np.full(10, 1500.0), age=np.full(10, 40), idx=np.full(10, 15), rated=rated)
    games = [(8, o, s) for o, s in zip(range(5), [1.0, 0.5, 0.5, 0.0, 1.0])] + [(9, o, s) for o, s in zip(range(3, 8), [1.0, 1.0, 0.0, 0.5, 0.5])] + [(8, 9, 1.0)]
    run_tournament(st, dr, m, games, part=np.r_[np.ones(10, bool), False])
    assert st.rated[8] and st.rated[9]
    # Rückrechnung: Der Fixpunkt der WO erfüllt We(Ri) = W für beide (vor der Korrektur), also stimmt die Korrektur nach Rückwärtsrechnung
    def undo(r1):
        return brentq(lambda ri: (ri + 0.08 * (2000 - ri)) - r1, 1100, 2000) if r1 >= 1172 else None
    ri8, ri9 = undo(st.R[8]), undo(st.R[9])
    assert ri8 is not None and ri9 is not None
    we8 = sum(ref_p(ri8 - rating[o]) for o in range(5)) + ref_p(ri8 - ri9)
    we9 = sum(ref_p(ri9 - rating[o]) for o in range(3, 8)) + ref_p(ri9 - ri8)
    assert we8 == pytest.approx(3.0 + 1.0, abs=0.02) and we9 == pytest.approx(3.0 + 0.0, abs=0.02)


# --- Bilanz -------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(), dict(churn=0.0), dict(gap=300.0, rate=0.2), dict(leave="die Schwächsten"), dict(trend=3.0), dict(sched="Aktivität und Mannschaftsturnier", ages=True),
                                dict(uplift_at=1), dict(anchor_join=1), dict(kmode="fester K", kfix=20.0, special=False, use_a=False, use_b=False, use_r=False, use_floor=False)])
def test_bilanz_der_wertungssumme_geht_exakt_auf(kw):
    import dwz_evaluation as ev
    for seed in range(4):
        run = F.simulate(small(n=14, seasons=4, **kw), 500 + seed)
        s = run.series
        total = s["sum_prev"] + s["anchor_in"] + s["uplift_gain"] + s["entry"] + s["play_a"] + s["play_b"] + s["play_special"] + s["play_rest"] + s["rounding"] + s["floor_lift"]
        assert np.abs(total - s["total_end"]).max() < 1e-8
        assert np.abs(s["total_end"] - s["leave"] - s["total_after"]).max() < 1e-8
        bal = ev.balance(run)
        assert bal["entry"] - bal["leave"] + bal["a"] + bal["b"] + bal["special"] + bal["rest"] + bal["rounding"] + bal["floor"] + bal["uplift"] + bal["anchor"] == pytest.approx(bal["end"], abs=1e-6)
