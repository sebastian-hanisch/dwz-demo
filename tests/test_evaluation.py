"""Kennzahlen, Analyse des gezeigten Laufs, Tabellen und Verdikte (an handgesetzten Werten und an den vorberechneten Experimenten)."""

import dataclasses

import numpy as np
import pytest

import dwz_constants as C
import dwz_evaluation as ev
from dwz_scenario import Settings


def small_settings(**kw) -> Settings:
    return Settings(**{"n": 20, "seasons": 6, "churn": 0.1, **kw})


def test_formatierung():
    assert ev.de(3.14159, 2) == "3,14" and ev.pct(0.15) == "15" and ev.thou(12345.6) == "12.346"
    assert ev.signed(5.4) == "+5" and ev.signed(-147.2) == "−147" and ev.signed(0.2) == "±0" and ev.signed(-0.04, 1) == "±0,0" and ev.signed(-0.4, 1) == "−0,4"


def test_analyse_hat_eine_laenge_je_saison_und_drift_ist_wertung_minus_faehigkeit():
    a = ev.analyse(small_settings())
    for arr in (a.drift, a.level, a.rating, a.slope, a.fit, a.anchor):
        assert len(arr) == 6
    assert np.allclose(a.drift, a.rating - a.level)


def test_bilanz_geht_auf_und_end_ist_die_aenderung_der_summe():
    a = ev.analyse(small_settings(use_uplift=True, uplift=2, use_anchor=True, join=1))
    b = a.bal
    assert b["entry"] - b["leave"] + b["a"] + b["b"] + b["special"] + b["rest"] + b["rounding"] + b["floor"] + b["uplift"] + b["anchor"] == pytest.approx(b["end"], abs=1e-6)
    assert b["uplift"] > 0 and b["anchor"] == C.BASE


def test_anker_ist_vor_dem_beitritt_leer_und_danach_gefuellt():
    a = ev.analyse(small_settings(use_anchor=True, join=2))
    assert np.isnan(a.anchor[:2]).all() and np.isfinite(a.anchor[2:]).all()
    assert np.isnan(ev.analyse(small_settings(use_anchor=False)).anchor).all()


def test_feldtabelle_zeigt_den_anker_nur_mit_anker_und_ungewertete_ohne_wertung():
    with_anchor = ev.field_table(ev.analyse(small_settings(use_anchor=True, join=1)))
    assert with_anchor[-1]["Spieler"] == "Anker" and with_anchor[-1]["Wertung"] is not None
    without = ev.field_table(ev.analyse(small_settings(use_anchor=False)))
    assert all(r["Spieler"] != "Anker" for r in without) and len(without) == 20
    unrated = [r for r in without if r["Wertung"] is None]
    assert all(r["Index"] == 0 and r["Abweichung"] is None for r in unrated)


def test_fehler_der_neuen_und_gehenden_sind_nan_im_geschlossenen_feld():
    a = ev.analyse(small_settings(churn=0.0))
    assert np.isnan(a.entry_err) and np.isnan(a.leave_err) and a.bal["entry"] == 0 and a.bal["leave"] == 0


def test_verdikt_des_laufs_schwellen_und_vorzeichen():
    a = ev.analyse(small_settings())
    low = dataclasses.replace(a, drift=np.array([0.0, 10.0]), rating=np.array([0.0, 12.0]), level=np.array([0.0, 2.0]), anchor=np.full(2, np.nan))
    level, text = ev.verdict_analysis(low)
    assert level == "info" and "Rauschen" in text and "+12" in text and "+2" in text
    up = dataclasses.replace(low, drift=np.array([0.0, 3 * C.NOISE]))
    assert ev.verdict_analysis(up)[0] == "warning" and "Inflation" in ev.verdict_analysis(up)[1] and f"+{round(3 * C.NOISE)}" in ev.verdict_analysis(up)[1]
    down = dataclasses.replace(low, drift=np.array([0.0, -3 * C.NOISE]), anchor=np.array([np.nan, 1450.0]))
    text = ev.verdict_analysis(down)[1]
    assert "Deflation" in text and "−50" in text and "Anker" in text


def test_beschreibung_der_experimente_zitiert_die_gemessenen_zahlen():
    def keys(group):
        return [k for k, g, _l, _kw in ev.catalog() if g == group]
    closed, youth = ev.rows(keys("closed") + keys("scen")), ev.rows(keys("youth"))
    by = {r["key"]: r for r in closed}
    text = ev.verdict_remedies(closed)[1]
    for k in ("c_all", "c_a", "c_special", "c_b", "c_floor"):
        assert ev.signed(by[k]["d40"]) in text
    ytext = ev.verdict_youth(youth)[1]
    by_y = {r["key"]: r for r in youth}
    for k in ("y_none", "y_all", "y_a", "y_r", "y_special", "y_noa"):
        assert ev.signed(by_y[k]["d40"]) in ytext
    assert ev.verdict_youth(youth)[0] == "warning" and "schießen über" in ytext
    btext = ev.verdict_balance(closed)[1]
    assert ev.signed(by["s_weak_none"]["entry_err"]) in btext and ev.signed(by["s_weak_none"]["leave_err"]) in btext
    stext = ev.verdict_special(closed)[1]
    assert ev.signed(by["s_learns_none"]["d40"]) in stext and ev.signed(by["s_learns_all"]["d40"]) in stext
    etext = ev.verdict_entry(closed)[1]
    assert ev.signed(by["s_random_none"]["entry_err"]) in etext and ev.signed(by["s_weak_none"]["entry_err"]) in etext


def test_erwartete_vorzeichen_der_experimente():
    """Die Befunde, auf denen die Texte stehen: Vorzeichen und Größenordnungen der vorberechneten Mittel."""
    m = lambda key: ev.mean(key, "d40")
    assert abs(m("c_none")) < 1 and abs(m("c_bare")) < 2                                   # ohne Gegenmittel steht das geschlossene Feld still
    assert m("c_all") > 100 and m("c_a") > 50 and m("c_special") > 30 and m("c_b") > 20
    assert m("y_none") < -100 and m("y_all") > 100 and m("y_a") > m("y_bare") + 100         # Deflation ohne, Überschuss mit den Regeln
    assert m("s_learns_none") < -50 and m("s_learns_all") > 100                              # die Sonderwertung dreht das Vorzeichen
    assert ev.mean("u_random_on", "direct_slope") < 0.75 and ev.series("u_random_on", "slope")[C.EXP_UPLIFT + 5] > 0.9
    assert abs(m("n_weak_table") - m("n_weak_base")) < 4 * ev.se("n_weak_base", "d40") and abs(m("n_closed_even") - m("n_closed_base")) < 1


def test_alle_verdikte_laufen_und_liefern_text():
    def keys(group):
        return [k for k, g, _l, _kw in ev.catalog() if g == group]
    cases = (ev.verdict_remedies(ev.rows(keys("closed") + keys("scen"))), ev.verdict_youth(ev.rows(keys("youth"))), ev.verdict_balance(ev.rows(keys("scen"))),
             ev.verdict_special(ev.rows(keys("closed") + keys("scen"))), ev.verdict_entry(ev.rows(keys("scen"))), ev.verdict_uplift(ev.rows(keys("uplift"))),
             ev.verdict_anchor(ev.rows(keys("anchor"))), ev.verdict_noeffect(ev.rows(keys("noeffect"))), ev.verdict_variants(ev.rows(keys("variants"))))
    for level, text in cases:
        assert level in ("info", "success", "warning") and len(text) > 100 and "nan" not in text.lower()
