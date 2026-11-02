"""Eigenschaftstests (metamorph) für die DWZ (ORACLE-PLAYBOOK §5.2): Man verändert die Eingabe auf eine Weise, deren Wirkung auf die Ausgabe aus dem Verfahren folgt, und prüft diese
Beziehung auf Zufallsinstanzen mit festem Seed. Die Orakel prüfen Rechenwege, hier stehen Aussagen, die für jede Einstellung gelten müssen:

- Bilanzidentität: Die Summe der Wertungen ändert sich nur durch die ausgewiesenen Posten (auf zufälligen Modellen, auch in `balance`);
- Verschiebung: Mit festem K ohne Zuschläge, Untergrenze und Sonderwertung ändert eine gemeinsame ganzzahlige Verschiebung aller Wertungen die Änderungen nicht (We hängt an Differenzen);
- Spiegelung: Alle Ergebnisse umkehren und die Wertungsachse spiegeln kehrt die Änderungen um (derselbe Rahmen);
- Umnummerieren und Umordnen: Plätze und Partien umsortieren ändert die Ergebnisse nur um die Umsortierung;
- Monotonie: Ein zusätzlicher Punkt in einer Partie hebt die eigene neue DWZ um K und senkt die des Gegners um dessen K (fester K, gleiche Erwartung);
- K0 ist mit dem Index nicht steigend und mit Alter und Wertung nicht steigend;
- Untergrenze: nie senkend und idempotent; Anhebung: monoton und nie senkend bis 2110;
- Erst-DWZ: gemeinsame Verschiebung der Gegner verschiebt die Leistung, Spiegelung kehrt sie um, mehr Punkte heben sie;
- Simulation: gleicher Seed gleiche Ausgabe, Läufe hängen nicht an Zahl und Reihenfolge der anderen.

Nicht geprüft: Verschiebung mit Zuschlägen und Sonderwertung (a, b, 1600, 2000, 1100 und die 300 hängen an absoluten Wertungen, dokumentiert)."""

import numpy as np
import pytest

import dwz_evaluation as ev
import dwz_field as F
import dwz_rules as W
from conftest import make_state, run_tournament, small

N_INSTANCES = 30
RNG_SEED = 20261102
PLAIN = dict(kmode="fester K", kfix=24.0, special=False, use_a=False, use_b=False, use_r=False, use_floor=False)


def _rng(salt: int) -> np.random.Generator:
    return np.random.default_rng(RNG_SEED + salt)


def _random_model(rng, **fixed) -> F.Model:
    n = int(rng.choice([8, 10, 12, 16]))
    kw = dict(n=n, seasons=int(rng.integers(2, 5)), burn=int(rng.integers(1, 4)), churn=float(rng.choice([0.0, 0.1, 0.25])), leave=str(rng.choice(["zufällig", "die Schwächsten"])),
              gap=float(rng.choice([0.0, 150.0, 350.0])), rate=float(rng.choice([0.1, 0.3])), trend=float(rng.choice([-3.0, 0.0, 4.0])), use_a=bool(rng.random() < 0.7),
              use_b=bool(rng.random() < 0.7), use_r=bool(rng.random() < 0.7), use_floor=bool(rng.random() < 0.7), special=bool(rng.random() < 0.7),
              sched=str(rng.choice(["zwei Turniere, alle spielen", "Aktivität und Mannschaftsturnier"])), ages=bool(rng.random() < 0.5))
    if rng.random() < 0.4:
        kw.update(kmode="fester K", kfix=float(rng.choice([16.0, 28.0])))
    if rng.random() < 0.5:
        kw["anchor_join"] = int(rng.integers(0, kw["seasons"] - 1)) if kw["seasons"] > 1 else None
    if rng.random() < 0.4:
        kw["uplift_at"] = int(rng.integers(0, kw["seasons"]))
    kw.update(fixed)
    return F.Model(**kw)


def _tournament_case(rng, n=10, plain=True, **kw):
    m = small(n=n, **(PLAIN if plain else {}), **kw)
    rating = rng.integers(1300, 2100, n).astype(float)
    st, dr = make_state(m, rating, rating.copy(), age=rng.integers(18, 60, n), idx=rng.integers(1, 15, n))
    games = []
    for _ in range(6):
        perm = rng.permutation(n)
        for k in range(n // 2):
            games.append((int(perm[2 * k]), int(perm[2 * k + 1]), float(rng.integers(0, 3)) / 2.0))
    return m, st, dr, games


# --- 1. Bilanzidentität ---------------------------------------------------------------------------------------------------------------------------

def test_die_wertungssumme_aendert_sich_nur_durch_die_ausgewiesenen_posten():
    rng = _rng(1)
    for _ in range(N_INSTANCES):
        m = _random_model(rng)
        run = F.simulate(m, int(rng.integers(1, 10 ** 6)))
        s = run.series
        total = s["sum_prev"] + s["anchor_in"] + s["uplift_gain"] + s["entry"] + s["play_a"] + s["play_b"] + s["play_special"] + s["play_rest"] + s["rounding"] + s["floor_lift"]
        assert np.abs(total - s["total_end"]).max() < 1e-7, m
        assert np.abs(s["total_end"] - s["leave"] - s["total_after"]).max() < 1e-7, m
        bal = ev.balance(run)
        assert sum(bal[k] * sign for k, sign in (("entry", 1), ("leave", -1), ("a", 1), ("b", 1), ("special", 1), ("rest", 1), ("rounding", 1), ("floor", 1), ("uplift", 1), ("anchor", 1))) == pytest.approx(bal["end"], abs=1e-6), m


def test_ohne_a_b_und_sonderwertung_gibt_es_keine_punkte_dieser_posten_und_im_geschlossenen_feld_mit_festem_k_keinen_rest():
    rng = _rng(2)
    for _ in range(N_INSTANCES):
        m = _random_model(rng, churn=0.0, use_a=False, use_b=False, special=False, kmode="fester K", kfix=20.0, use_floor=False, anchor_join=None, uplift_at=None, trend=0.0)
        run = F.simulate(m, int(rng.integers(1, 10 ** 6)))
        s = run.series
        assert np.abs(s["play_a"]).max() == np.abs(s["play_b"]).max() == np.abs(s["play_special"]).max() == 0
        assert np.abs(s["play_rest"]).max() < 1e-7 and np.abs(s["entry"]).max() == 0 and np.abs(s["leave"]).max() == 0


# --- 2. Verschiebung, Spiegelung, Umnummerieren ---------------------------------------------------------------------------------------------------

def test_gemeinsame_ganzzahlige_verschiebung_aendert_die_aenderungen_nicht():
    rng = _rng(3)
    for _ in range(N_INSTANCES):
        m, st, dr, games = _tournament_case(rng)
        before = st.R[:10].copy()
        run_tournament(st, dr, m, games, part=np.r_[np.ones(10, bool), False])
        base = st.R[:10] - before
        shift = float(rng.integers(-500, 1500))
        st2, dr2 = make_state(m, before + shift, before + shift, age=st.age[:10].copy() - 0, idx=np.array(st.idx[:10]) - 1)
        run_tournament(st2, dr2, m, games, part=np.r_[np.ones(10, bool), False])
        assert (st2.R[:10] - (before + shift) == base).all()


def test_spiegeln_der_ergebnisse_und_der_wertungsachse_kehrt_die_aenderungen_um():
    rng = _rng(4)
    for _ in range(N_INSTANCES):
        m, st, dr, games = _tournament_case(rng)
        before = st.R[:10].copy()
        run_tournament(st, dr, m, games, part=np.r_[np.ones(10, bool), False])
        change = st.R[:10] - before
        c = 4000.0
        mirrored = [(i, j, 1.0 - w) for i, j, w in games]
        st2, dr2 = make_state(m, c - before, c - before, age=np.array(st.age[:10]), idx=np.array(st.idx[:10]) - 1)
        run_tournament(st2, dr2, m, mirrored, part=np.r_[np.ones(10, bool), False])
        change2 = st2.R[:10] - (c - before)
        # Rundung bei x,5 ist nicht spiegelsymmetrisch: höchstens 1 Punkt Abweichung, sonst exakt
        assert np.abs(change2 + change).max() <= 1.0


def test_umnummerieren_der_spieler_und_umordnen_der_partien_aendert_nur_die_reihenfolge():
    rng = _rng(5)
    for _ in range(N_INSTANCES):
        m, st, dr, games = _tournament_case(rng, plain=False)
        rating, age, idx = st.R[:10].copy(), st.age[:10].copy(), st.idx[:10].copy()
        run_tournament(st, dr, m, games, part=np.r_[np.ones(10, bool), False])
        perm = rng.permutation(10)                                           # alt i → neu perm[i]
        inv = np.argsort(perm)
        st2, dr2 = make_state(m, rating[inv], rating[inv], age=age[inv], idx=idx[inv])
        shuffled = [games[k] for k in rng.permutation(len(games))]
        relabelled = [(int(perm[i]), int(perm[j]), w) for i, j, w in shuffled]
        run_tournament(st2, dr2, m, relabelled, part=np.r_[np.ones(10, bool), False])
        assert (st2.R[:10] == st.R[:10][inv]).all()


def test_ein_zusaetzlicher_punkt_hebt_die_eigene_dwz_um_k_und_senkt_die_des_gegners():
    rng = _rng(6)
    for _ in range(N_INSTANCES):
        n = 8
        m = small(n=n, **PLAIN)
        rating = rng.integers(1300, 2100, n).astype(float)
        base = [(0, 1, 0.0), (2, 3, 0.5), (4, 5, 1.0), (6, 7, 0.0)]
        better = [(0, 1, 1.0)] + base[1:]
        out = []
        for games in (base, better):
            st, dr = make_state(m, rating, rating.copy(), age=np.full(n, 40), idx=np.full(n, 15))
            run_tournament(st, dr, m, games, part=np.r_[np.ones(n, bool), False])
            out.append(st.R[:2].copy())
        assert out[1][0] - out[0][0] == pytest.approx(24.0, abs=1.0) and out[0][1] - out[1][1] == pytest.approx(24.0, abs=1.0)


# --- 3. Regeln ------------------------------------------------------------------------------------------------------------------------------------

def test_grundwert_k0_ist_im_index_und_in_der_gruppe_nicht_steigend():
    t = W.K0_TABLE
    assert (np.diff(t, axis=1) <= 0).all()                                   # Index: nie größer
    assert (t[0] >= t[1]).all() and (t[1] >= t[2]).all()                     # jung ≥ alt ≥ stark (gleicher Index)


def test_k_faktor_ist_im_index_nicht_steigend_bei_sonst_gleichen_angaben():
    rng = _rng(7)
    for _ in range(N_INSTANCES):
        r0, age, win = float(rng.integers(1000, 2600)), int(rng.integers(10, 70)), bool(rng.integers(0, 2))
        ks = [float(W.k_factor(np.array([r0]), np.array([age]), np.array([i]), np.array([win]))[0][0]) for i in range(1, 14)]
        assert all(a >= b for a, b in zip(ks, ks[1:]))


def test_untergrenze_senkt_nie_und_ist_idempotent():
    rng = _rng(8)
    r0, k, w, we = rng.integers(1000, 1300, 300).astype(float), rng.integers(10, 80, 300).astype(float), rng.integers(0, 10, 300) / 2.0, rng.random(300) * 9
    with_floor = W.update(r0, k, w, we)[0]
    without = W.update(r0, k, w, we, floor=None)[0]
    assert (with_floor >= without).all() and (with_floor >= W.FLOOR).all()
    again = W.update(with_floor, np.zeros(300), w, we)[0]
    assert (again == with_floor).all()


def test_anhebung_ist_monoton_und_hebt_nie_ab():
    r = np.sort(_rng(9).integers(0, 3200, 500).astype(float))
    u = W.uplift(r)
    assert (np.diff(u) >= 0).all() and (u >= r).all()


def test_erst_dwz_verschiebt_sich_mit_den_gegnern_und_spiegelt_mit_den_ergebnissen():
    rng = _rng(10)
    for _ in range(N_INSTANCES):
        n = int(rng.integers(5, 9))
        opp = rng.integers(1300, 1900, n).astype(float)
        sc = rng.integers(0, 3, n) / 2.0
        w = float(sc.sum())
        if w <= 0 or w >= n:
            continue
        zeros = np.zeros(n, dtype=int)
        base = float(W.performance(zeros, opp, sc, 1, np.array([False]))[0])
        c = float(rng.integers(100, 500))
        assert float(W.performance(zeros, opp + c, sc, 1, np.array([False]))[0]) == pytest.approx(base + c, abs=1e-3)
        assert float(W.performance(zeros, 6000.0 - opp, 1.0 - sc, 1, np.array([False]))[0]) == pytest.approx(6000.0 - base, abs=1e-3)
        better = sc.copy()
        better[np.argmin(sc)] = 1.0
        assert float(W.performance(zeros, opp, better, 1, np.array([False]))[0]) >= base


def test_erst_dwz_korrektur_hebt_unter_2000_an_und_laesst_darueber_unveraendert():
    ri = np.arange(1100.0, 2600.0, 7.0)
    r1 = W.first_rating_correction(ri)
    assert (r1[ri < 2000] > ri[ri < 2000]).all() and (r1[ri >= 2000] == ri[ri >= 2000]).all()


# --- 4. Simulation --------------------------------------------------------------------------------------------------------------------------------

def test_gleicher_seed_gleiche_ausgabe_und_laeufe_sind_unabhaengig():
    rng = _rng(11)
    for _ in range(10):
        m = _random_model(rng)
        seed = int(rng.integers(1, 10 ** 6))
        a = F.simulate(m, seed)
        F.simulate(m, seed + 1)
        b = F.simulate(m, seed)
        for f in F.SEASON_FIELDS:
            assert np.array_equal(a.series[f], b.series[f], equal_nan=True)


def test_mehr_gegenmittel_geben_im_geschlossenen_feld_mehr_punkte():
    """Jede zusätzliche Regel (a, b, Sonderwertung) zahlt im Mittel Punkte aus, keine nimmt sie weg (60 Läufe, geschlossenes Feld)."""
    base = dict(n=40, seasons=12, burn=4, churn=0.0, gap=0.0)
    seeds = range(100, 124)
    def drift(**kw):
        return float(np.mean([ev.summarize(F.simulate(F.Model(**base, **kw), s))["d40"] for s in seeds]))
    bare = drift(use_a=False, use_b=False, special=False, use_floor=False, use_r=False)
    for kw in (dict(use_a=True, use_b=False, special=False), dict(use_a=False, use_b=True, special=False), dict(use_a=False, use_b=False, special=True)):
        assert drift(use_floor=False, use_r=False, **kw) > bare + 2.0, kw
