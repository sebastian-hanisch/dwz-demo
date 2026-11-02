"""Einheitentests der Regeln (dwz_rules): Handwerte und Grenzfälle, an der WO nachgerechnet."""

import numpy as np
import pytest

import dwz_rules as W


def k(r0, age, idx, win, **kw):
    out = W.k_factor(np.array([float(r0)]), np.array([age]), np.array([idx]), np.array([win]), **kw)
    return float(out[0][0])


def test_gewinnerwartung_symmetrisch_und_monoton():
    d = np.linspace(-1000, 1000, 201)
    p = W.win_prob(d)
    assert np.allclose(p + p[::-1], 1.0) and (np.diff(p) > 0).all()
    assert float(W.win_prob(0.0)) == 0.5 and float(W.win_prob(200.0)) == pytest.approx(0.760, abs=0.001)


def test_tabellenfassung_rundet_d_ganzzahlig_und_p_auf_drei_stellen():
    assert float(W.win_prob(100.4, table=True)) == float(W.win_prob(100.0, table=True)) == 0.638
    assert float(W.win_prob(100.6, table=True)) == float(W.win_prob(101.0, table=True))


def test_umkehrung_ist_auf_729_begrenzt():
    assert float(W.rating_difference(0.5)) == 0.0
    assert float(W.rating_difference(0.999)) == pytest.approx(float(W.rating_difference(0.995))) == pytest.approx(728.5, abs=0.5)
    assert float(W.rating_difference(0.001)) == pytest.approx(-728.5, abs=0.5)
    assert float(W.rating_difference(W.win_prob(150.0))) == pytest.approx(150.0)


@pytest.mark.parametrize("r0,age,idx,win,expect", [
    (1650, 40, 13, True, 28.0),             # Erwachsener unter 2000, Index > 10, Rechner-Beispiel
    (1300, 40, 13, False, 24.5),            # Bremsfaktor 0,875
    (1300, 40, 13, True, 32.0),             # unter 1600 mit Sieg: 28 + 4
    (1650, 15, 13, True, 43.7),             # Jugendlicher: 32 + (2000 − 1650)/30 = 43,667
    (1650, 15, 13, False, 32.0),            # kein Aufschlag bei Verlust
    (2200, 25, 3, False, 41.0),             # Junior ab 2200: dritte Zeile
    (2199, 25, 3, False, 48.0),             # … knapp darunter: erste Zeile
    (2000, 26, 3, False, 41.0), (1999, 26, 3, False, 44.0),
    (1599, 40, 20, True, 32.0), (1600, 40, 20, True, 28.0),     # Grenze 1600 für a bei Erwachsenen
    (1999, 20, 20, True, 32 + 1 / 30), (2000, 20, 20, True, 32.0), (2200, 20, 20, True, 25.0),
    (1500, 21, 20, True, 36.0),             # ab 21 Jahren gilt a = 4 statt der Jugend-Formel (Junior: 32 + 4)
    (1500, 20, 20, True, 48.7),             # bis 20 Jahre: (2000 − 1500)/30 = 16,667
])
def test_k_faktor_handwerte_und_grenzen(r0, age, idx, win, expect):
    assert k(r0, age, idx, win) == pytest.approx(round(expect, 1), abs=1e-9)


def test_k_faktor_ist_auf_kmax_begrenzt():
    assert k(1100, 15, 1, True) == 80.0                      # 60 + (2000 − 1100)/30 = 90 → 80


def test_index_oberhalb_zehn_zaehlt_wie_elf():
    assert k(1700, 40, 11, False) == k(1700, 40, 99, False) == 28.0
    assert k(1700, 40, 10, False) == 30.0 and k(1700, 40, 1, False) == k(1700, 40, 2, False) == 60.0


def test_fester_k_ignoriert_alter_index_und_zuschlaege():
    out = W.k_factor(np.array([1300.0, 2300.0]), np.array([15, 60]), np.array([1, 40]), np.array([True, False]), fixed=20.0)
    assert list(out[0]) == [20.0, 20.0] and (out[2] == 0).all() and (out[3] == 1).all()


def test_aufschlag_und_bremse_lassen_sich_einzeln_abschalten():
    assert k(1300, 40, 13, True, use_a=False) == 28.0 and k(1300, 40, 13, False, use_b=False) == 28.0


def test_folge_dwz_untergrenze_und_rundung():
    rn, raw, rounded = W.update(np.array([1105.0, 1500.0, 1500.0]), np.array([40.0, 20.0, 20.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.5, 0.03]))
    assert list(rn) == [1100.0, 1510.0, 1499.0] and raw[0] == pytest.approx(1065.0) and rounded[0] == 1065.0
    # kaufmännisch gegen numpy-Standard bei x,5
    assert float(W.round_rating(2.5)) == 3.0 and float(W.round_rating(2.5, "even")) == 2.0 and float(W.round_rating(-0.5)) == 0.0
    assert float(W.update(np.array([1000.0]), np.array([20.0]), np.array([1.0]), np.array([0.0]), floor=None)[0][0]) == 1020.0


def test_erst_dwz_korrektur_ist_stetig_und_nie_unter_1100():
    ri = np.linspace(-200, 2600, 561)
    r1 = W.first_rating_correction(ri)
    assert (r1 >= 1100).all() and (np.diff(r1) >= -1e-9).all()
    assert float(W.first_rating_correction(np.array([1100.0]))[0]) == pytest.approx(1172.0)
    assert float(W.first_rating_correction(np.array([0.0]))[0]) == 1100.0 and float(W.first_rating_correction(np.array([1500.0]), 0.0)[0]) == 1500.0


def test_leistung_bei_null_punkten_wird_null_und_ist_nie_negativ():
    opp = np.array([1500.0] * 6)
    assert float(W.performance(np.zeros(6, dtype=int), opp, np.zeros(6), 1, np.array([False]))[0]) == 0.0


def test_leistung_bei_gleichem_ergebnis_ist_der_gegnerschnitt():
    opp = np.array([1400.0, 1600.0, 1500.0, 1450.0, 1550.0, 1500.0])
    got = float(W.performance(np.zeros(6, dtype=int), opp, np.full(6, .5), 1, np.array([False]))[0])
    assert got == pytest.approx(1500.0, abs=0.01)


def test_iteration_eines_einzelnen_zusatzspielers_ist_seine_leistung():
    opp = np.array([1400.0, 1600.0, 1500.0, 1450.0, 1550.0, 1650.0])
    sc = np.array([1, 0, .5, 1, 0, .5])
    perf = float(W.performance(np.zeros(6, dtype=int), opp, sc, 1, np.array([False]))[0])
    x, step = W.total_iteration(np.zeros(6, dtype=int), np.full(6, -1), opp, sc, np.array([True]), 1, tol=1e-9, max_iter=2000)
    assert float(x[0]) == pytest.approx(perf, abs=1e-3) and step < 1e-8


def test_anhebung_ist_monoton_und_hebt_bis_2110_an():
    r = np.arange(0.0, 3000.0)
    u = W.uplift(r)
    assert (np.diff(u) >= 0).all() and (u[:2111] >= r[:2111]).all() and (u[2112:] == r[2112:]).all()
    assert float(W.uplift(np.array([1000.0]))[0]) == 1400.0 and float(W.uplift(np.array([1001.0]))[0]) == 1401.0       # 0,64·1001 + 760 = 1400,64: stetig
    assert float(W.uplift(np.array([2110.0]))[0]) - 2110.0 == 0.0                                                    # 2110,4 gerundet
    assert float(W.uplift(np.array([2111.0]))[0]) == 2111.0
