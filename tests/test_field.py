"""Einheitentests des Feldmodells (dwz_field): Zufallsstrom, Partien, Abgangsregeln, Anker, Anhebung, geschlossenes Feld."""

import numpy as np
import pytest

import dwz_constants as C
import dwz_field as F
from conftest import make_state, small


def test_zufallsstrom_hat_feste_groesse_und_haengt_nur_am_seed():
    a, b = F.draws(5, 10, 13), F.draws(5, 10, 13)
    c = F.draws(6, 10, 13)
    for key in a:
        assert (a[key] == b[key]).all() and not (a[key] == c[key]).all()
    assert a["pair"].shape == (10, F.TOURNAMENTS, F.ROUNDS, 13) and a["z"].shape == (11, 13) and a["leave"].shape == (10, 13)
    assert ((a["pair"] >= 0) & (a["pair"] < 1)).all()


def test_schalter_der_regeln_aendern_den_zufallsstrom_nicht():
    """Gepaarte Vergleiche: dieselben Gründerstärken und Paarungsschlüssel für jede Regelvariante."""
    dr = F.draws(3, small().total, small().slots)
    base = F._State(small(), dr)
    for kw in (dict(use_a=False), dict(special=False), dict(kmode="fester K"), dict(sched=C.SCHED[1]), dict(use_floor=False)):
        other = F._State(small(**kw), dr)
        assert (other.true == base.true).all() and (other.R == base.R).all()
    assert np.array_equal(F.draws(3, small().total, small().slots)["pair"], dr["pair"])


def test_modell_prueft_seine_eingaben():
    with pytest.raises(ValueError):
        F.simulate(small(n=3), 1)
    with pytest.raises(ValueError):
        F.simulate(small(churn=0.99), 1)
    with pytest.raises(ValueError):
        F.simulate(small(anchor_join=3), 1)                  # seasons = 3: nur 0 bis 2
    with pytest.raises(ValueError):
        F.simulate(small(uplift_at=5), 1)
    with pytest.raises(ValueError):
        F.leavers(small(leave="unbekannt"), np.ones(13, bool), np.zeros(13), np.zeros(13))


def test_lauf_ist_deterministisch_und_unabhaengig_von_der_reihenfolge():
    a, b = F.simulate(small(), 11), F.simulate(small(), 12)
    a2, b2 = F.simulate(small(), 11), F.simulate(small(), 12)
    for f in F.SEASON_FIELDS:
        assert np.array_equal(a.series[f], a2.series[f], equal_nan=True) and np.array_equal(b.series[f], b2.series[f], equal_nan=True)
    assert not np.array_equal(a.series["bias"], b.series["bias"])


def test_abgangsregeln():
    m = small(n=6, churn=0.5)
    rated = np.array([True, True, True, True, False, False, False])
    rating = np.array([1700.0, 1400, 1600, 1300, 0, 0, 0])
    key = np.array([0.9, 0.1, 0.5, 0.3, 0.2, 0.8, 0.0])
    rnd = F.leavers(small(n=6, churn=0.5, leave=C.LEAVE[0]), rated, rating, key)
    assert list(rnd) == [1, 4, 3]                                           # die kleinsten Schlüssel der Plätze 0 bis 5: 0,1 (1), 0,2 (4), 0,3 (3)
    weak = F.leavers(small(n=6, churn=0.5, leave=C.LEAVE[1]), rated, rating, key)
    assert list(weak) == [4, 5, 3]                                          # Ungewertete (4, 5) zuerst, dann die niedrigste Wertung (Platz 3: 1300)
    assert len(F.leavers(m, rated, rating, key)) == m.n_out == 3


def test_partien_eines_turniers_paarweise_gespiegelt_und_nur_unter_teilnehmern():
    m = small()
    st, dr = make_state(m, np.full(12, 1500.0), np.linspace(1300, 1700, 12))
    part = np.zeros(m.slots, dtype=bool)
    part[[0, 1, 2, 3, 4, 5, 6]] = True                                     # 7 Teilnehmer: einer setzt je Runde aus
    P, O, S = F._games(st, dr, 0, 0, 9, part)
    assert len(P) == len(O) == len(S) == 2 * 3 * 9
    assert set(P) <= set(range(7)) and set(O) <= set(range(7)) and (P != O).all()
    assert set(np.unique(S)) <= {0.0, 0.5, 1.0}
    for r in range(9):                                                      # je Runde stehen die 3 Partien zuerst aus der einen, dann aus der anderen Sicht
        blk = slice(6 * r, 6 * r + 6)
        assert np.allclose(S[blk][:3] + S[blk][3:], 1.0) and (P[blk][:3] == O[blk][3:]).all() and (O[blk][:3] == P[blk][3:]).all()
    assert max(np.bincount(P, minlength=7)) <= 9


def test_punkterwartung_der_simulierten_partien_ist_die_gewinnerwartung():
    m = small(n=4)
    st, dr = make_state(m, np.full(4, 1500.0), np.array([1500.0, 1700.0, 1500.0, 1700.0]))
    import dwz_rules as W
    total, games = 0.0, 0
    for seed in range(300):
        dr = F.draws(seed, m.total, m.slots)
        P, O, S = F._games(st, dr, 0, 0, 9, np.r_[np.ones(4, bool), False])
        mask = (P == 0) & (O == 1)
        total += S[mask].sum()
        games += mask.sum()
    assert games > 300 and total / games == pytest.approx(float(W.win_prob(-200.0)), abs=0.025)


def test_geschlossenes_feld_mit_festem_k_ohne_regeln_erhaelt_die_summe_bis_auf_rundung():
    m = small(n=14, seasons=4, churn=0.0, kmode="fester K", kfix=24.0, special=False, use_a=False, use_b=False, use_r=False, use_floor=False)
    run = F.simulate(m, 21)
    s = run.series
    assert np.abs(s["play_a"]).max() == 0 and np.abs(s["play_b"]).max() == 0 and np.abs(s["play_special"]).max() == 0
    assert np.abs(s["play_rest"]).max() < 1e-8                              # gleiches K für alle: die Partien zahlen nichts aus
    assert np.abs(s["total_end"] - s["sum_prev"] - s["rounding"]).max() < 1e-8
    assert np.abs(s["rounding"]).max() < 14                                 # nur Rundungsreste, höchstens ein halber Punkt je Spieler und Turnier


def test_anker_tritt_bei_bleibt_bei_1500_und_geht_nie():
    run = F.simulate(small(seasons=5, anchor_join=2), 4)
    s = run.series
    assert np.isnan(s["anchor"][:2 + 2]).all() and np.isfinite(s["anchor"][2 + 2:]).all()      # burn = 2: Beitritt zu Beginn von Saison 4
    assert s["anchor_in"][4] == C.BASE and s["anchor_in"].sum() == C.BASE
    assert run.final["true"][12] == C.BASE and run.final["active"][12] and run.final["rated"][12]


def test_ohne_anker_bleibt_der_platz_leer():
    run = F.simulate(small(), 4)
    assert not run.final["active"][12] and np.isnan(run.series["anchor"]).all()


def test_anhebung_wird_zu_beginn_der_saison_angewendet_und_gemerkt():
    run = F.simulate(small(n=20, churn=0.1, seasons=4, uplift_at=2), 9)
    s = run.series
    assert s["uplift_gain"][4] > 0 and s["uplift_gain"].sum() == s["uplift_gain"][4]
    assert np.isfinite(run.direct).all() and run.direct[1] < 0.85          # Steigung unmittelbar nach der Anhebung: gestaucht
    assert np.isnan(F.simulate(small(n=20, churn=0.1, seasons=4), 9).direct).all()


def test_neue_kommen_ungewertet_und_bekommen_nach_einem_turnier_eine_erst_dwz():
    run = F.simulate(small(n=20, seasons=4, churn=0.25, burn=3), 6)
    s = run.series
    assert s["entry_n"][:3].sum() == 0                                      # Einlaufzeit: niemand wird neu gewertet
    assert s["entry_n"][3:].sum() >= 5 and s["leave_n"].sum() > 0
    assert (s["entry_n"][3:] <= 5 + 5).all()


def test_jugend_startet_unter_ihrer_spitze_und_holt_auf():
    m = small(n=10, seasons=3, gap=300.0, rate=0.5)
    dr = F.draws(2, m.total, m.slots)
    st = F._State(m, dr)
    st.new_player(0, 0, dr, 1500.0)
    assert st.true[0] == pytest.approx(st.pot[0] - 300.0) and st.age[0] == 14 and st.youth[0] and not st.rated[0] and st.idx[0] == 0
    adult = F._State(small(n=10, gap=0.0), dr)
    adult.new_player(0, 0, dr, 1500.0)
    assert adult.age[0] == 30 and not adult.youth[0] and adult.true[0] == adult.pot[0]


def test_zeitplaene():
    assert F.SCHEDULES[C.SCHED[0]] == ((9, "all"), (9, "all")) and len(F.SCHEDULES[C.SCHED[1]]) == 3 and F.SCHEDULES[C.SCHED[1]][2] == (7, "team")
