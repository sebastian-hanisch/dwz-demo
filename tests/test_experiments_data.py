"""Die vorberechneten Experimente (data/experiments.json) gehören zum heutigen Code: Jede Konfiguration des Katalogs steht in der Datei, und je Konfiguration stimmen die Läufe,
die hier frisch gerechnet werden, mit der Datei überein. Ändert sich Regel oder Feldmodell, schlägt dieser Test fehl, bis `python tools/build_experiments.py` neu gelaufen ist."""

import numpy as np
import pytest

import dwz_constants as C
import dwz_evaluation as ev


def test_der_katalog_hat_vierundfuenfzig_konfigurationen_mit_eindeutigen_schluesseln():
    keys = [k for k, _g, _l, _kw in ev.catalog()]
    assert len(keys) == len(set(keys)) == 54


def test_die_datei_enthaelt_genau_den_katalog_und_sechzig_laeufe_je_konfiguration():
    data = ev.load_data()
    assert data["seeds"] == list(C.SWEEP_SEEDS) and set(data["configs"]) == {k for k, *_ in ev.catalog()}
    for key, c in data["configs"].items():
        assert c["seeds"] == list(C.SWEEP_SEEDS), key
        for field in ev.RUN_FIELDS:
            assert len(c["runs"][field]) == 60, (key, field)
        for name, ys in c["series"].items():
            assert len(ys) == C.EXP_SEASONS, (key, name)


@pytest.mark.parametrize("key", [k for k, *_ in ev.catalog()])
def test_ein_lauf_je_konfiguration_stimmt_mit_der_datei_ueberein(key):
    fresh = ev.compute_config(key, seeds=C.SWEEP_SEEDS[:1])
    stored = ev.load_data()["configs"][key]
    for field in ev.RUN_FIELDS:
        a, b = fresh["runs"][field][0], stored["runs"][field][0]
        assert (a is None and b is None) or a == pytest.approx(b, abs=1e-9), (key, field, a, b)


def test_zeilen_der_experimente_haben_alle_felder():
    rows = ev.rows(["c_all", "u_random_on", "a_closed_2"])
    for r in rows:
        for f in ("d10", "d20", "d40", "se", "sd", "level", "rating", "slope", "fit", "series", "entry", "leave", "a", "b", "special", "rest", "rounding", "floor", "uplift", "anchor_in",
                  "cases", "anchor4", "anchor4_se", "direct_slope", "direct_fit", "entry_err", "leave_err"):
            assert f in r, (r["key"], f)
    assert len(rows[0]["series"]) == C.EXP_SEASONS and np.isnan(rows[0]["anchor4"]) and np.isfinite(rows[2]["anchor4"]) and np.isfinite(rows[1]["direct_slope"])


def test_standardfehler_und_mittel_sind_aus_den_laeufen_gerechnet():
    xs = np.array(ev.cfg("c_all")["runs"]["d40"], dtype=float)
    assert ev.mean("c_all", "d40") == pytest.approx(xs.mean()) and ev.se("c_all", "d40") == pytest.approx(xs.std(ddof=1) / np.sqrt(60)) and ev.sd("c_all", "d40") == pytest.approx(xs.std(ddof=1))
