"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed). Die Experimente sind in
data/experiments.json vorberechnet; `test_experiments_data.py` stellt sicher, dass die Datei zum Code gehört. Der Zufallsstrom hängt nur von Seed und Position ab (SplitMix64), die
Zahlen sind plattformunabhängig. Ganze Punkte gelten mit ±0,6 (Rundung der README-Angabe)."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import dwz_constants as C
import dwz_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def d40(key):
    return ev.mean(key, "d40")


def near(x, want, tol=0.6):
    return x == pytest.approx(want, abs=tol)


# --- Experiment 1: Was die Gegenmittel erzeugen -----------------------------------------------------------------------------------------------------

def test_a_closed_field_without_remedies_and_with_only_k_from_index_and_age_stands_still():
    assert abs(d40("c_none")) < 1.0 and abs(d40("c_bare")) < 2.0


def test_all_rules_pay_out_154_points_in_a_closed_field_and_a_pays_most():
    assert near(d40("c_all"), 154) and ev.se("c_all", "d40") < 2.0
    assert near(d40("c_a"), 85) and near(d40("c_special"), 70) and near(d40("c_b"), 42) and near(d40("c_floor"), 18)
    assert d40("c_a") > d40("c_special") > d40("c_b") > d40("c_floor") > 0


def test_the_points_of_the_rules_are_the_posts_of_the_balance():
    assert near(ev.mean("c_all", "a") / C.EXP_N, 64, 1.0) and near(ev.mean("c_all", "special") / C.EXP_N, 68, 1.0)
    assert near(ev.mean("c_all", "cases"), 54, 0.6)


# --- Experiment 2: Lernende Jugend ----------------------------------------------------------------------------------------------------------------

def test_learning_youth_deflates_without_remedies_and_inflates_with_all_rules():
    assert near(d40("y_none"), -130) and near(d40("y_all"), 180) and near(d40("y_bare"), -150)
    assert d40("y_none") < -100 < 100 < d40("y_all")


def test_the_success_surcharge_a_does_most_of_the_work_for_the_youth():
    assert near(d40("y_a"), -3) and near(d40("y_noa"), 49) and near(d40("y_r"), -43) and near(d40("y_special"), -140)
    assert d40("y_a") - d40("y_bare") > 140 and d40("y_all") - d40("y_noa") > 120


# --- Experiment 3: Bilanz und Eintrittsregel ------------------------------------------------------------------------------------------------------

def rows(*keys):
    return {r["key"]: r for r in ev.rows(list(keys))}


def test_without_remedies_the_drift_is_entry_minus_exit():
    r = rows("s_random_none", "s_weak_none")
    assert near(r["s_random_none"]["entry_err"], 16) and near(r["s_random_none"]["leave_err"], 12) and near(r["s_random_none"]["d40"], 25)
    assert near(r["s_weak_none"]["entry_err"], 88) and near(r["s_weak_none"]["leave_err"], 57, 1.0) and near(r["s_weak_none"]["d40"], 172)
    # Abgangsquote mal Differenz der Fehler, 40 Saisons: Größenordnung der gemessenen Drift
    expect = C.EXP_CHURN * (r["s_weak_none"]["entry_err"] - r["s_weak_none"]["leave_err"]) * C.EXP_SEASONS
    assert 0.8 * r["s_weak_none"]["d40"] < expect < 1.3 * r["s_weak_none"]["d40"]


def test_with_all_rules_both_scenarios_end_near_plus_300():
    r = rows("s_random_all", "s_weak_all", "s_youth_all", "s_learns_all")
    assert near(r["s_random_all"]["d40"], 303) and near(r["s_weak_all"]["d40"], 301) and near(r["s_learns_all"]["d40"], 163)
    assert near(r["s_random_all"]["entry_err"], 247) and near(r["s_weak_all"]["entry_err"], 246)


def test_the_balance_posts_add_up_to_the_change_of_the_sum_in_every_experiment_configuration():
    for key, group, _label, _kw in ev.catalog():
        if group == "anchor":
            continue
        r = ev.rows([key])[0]
        end = r["entry"] - r["leave"] + r["a"] + r["b"] + r["special"] + r["rest"] + r["rounding"] + r["floor"] + r["uplift"] + r["anchor_in"]
        assert end == pytest.approx(ev.mean(key, "entry") - ev.mean(key, "leave") + ev.mean(key, "a") + ev.mean(key, "b") + ev.mean(key, "special") + ev.mean(key, "rest")
                                    + ev.mean(key, "rounding") + ev.mean(key, "floor") + ev.mean(key, "uplift") + ev.mean(key, "anchor_in"), abs=1e-6)


def _entry_error(gap, threshold, r, reps=20000, games=9, field=60, seed=7):
    """Kontrollversuch: ein Neuling mit Stärke Feldmittel + gap spielt 9 Partien gegen ein genau gewertetes Feld (Mittel 1700, Streuung 200); mittlerer Fehler der Erst-DWZ der Gewerteten."""
    import numpy as np
    import dwz_rules as W
    rng = np.random.default_rng(seed)
    fr = rng.normal(1700.0, 200.0, (reps, field))
    idx = np.argsort(rng.random((reps, field)), axis=1)[:, :games]
    opp = np.take_along_axis(fr, idx, axis=1)
    t = fr.mean(axis=1) + gap
    w = (rng.random((reps, games)) < W.win_prob(t[:, None] - opp)).sum(axis=1).astype(float)
    rated = w >= threshold
    x = np.empty(reps)
    full = w >= games
    for i in np.nonzero(rated)[0]:
        x[i] = float(W.performance(np.zeros(games, dtype=int), opp[i], np.r_[np.ones(int(w[i])), np.zeros(games - int(w[i]))], 1, np.array([bool(full[i])]))[0])
    r1 = W.first_rating_correction(x[rated], r)
    return float(np.mean(r1 - t[rated])), float(rated.mean())


def test_the_threshold_of_half_a_point_selects_the_lucky_newcomers():
    """Kontrollversuch zur Eintrittsregel (Experiment 5): Je weiter ein Neuling unter dem Feld liegt, desto stärker überschätzt ihn die Erst-DWZ (ohne den Aufschlag r), wenn nur gewertet wird,
    wer mindestens 0,5 Punkte holt; ohne die Hürde (jede Erst-DWZ) liegt der Fehler deutlich tiefer."""
    near_err, near_p = _entry_error(-100, 0.5, 0.0, reps=3000)
    far_err, far_p = _entry_error(-300, 0.5, 0.0, reps=3000)
    farthest_err, farthest_p = _entry_error(-500, 0.5, 0.0, reps=3000)
    assert near(near_err, -2, 8.0) and near(far_err, 18, 8.0) and near(farthest_err, 116, 15.0)        # 100 / 300 / 500 Punkte unter dem Feld
    assert near(near_p, 0.99, 0.02) and near(far_p, 0.86, 0.03) and near(farthest_p, 0.50, 0.04)       # Anteil der Gewerteten
    assert near(_entry_error(-300, 0.0, 0.0, reps=3000)[0], -27, 8.0)                                  # ohne Hürde: unter der Stärke statt darüber
    assert near(_entry_error(-300, 0.5, 0.08, reps=3000)[0], 64, 8.0)                                  # mit dem Aufschlag r = 0,08 der WO


# --- Experiment 4: Sonderwertung -------------------------------------------------------------------------------------------------------------------

def test_the_special_rating_turns_the_sign_when_the_whole_field_learns():
    assert near(d40("s_learns_none"), -115) and near(d40("s_learns_all"), 163)
    assert near(d40("c_special"), 70) and abs(d40("c_bare")) < 2.0


# --- Experiment 6: Anhebung ------------------------------------------------------------------------------------------------------------------------

def test_the_uplift_compresses_the_scale_and_the_fixed_expectation_stretches_it_back():
    u = C.EXP_UPLIFT
    off, on = ev.series("u_random_off", "slope"), ev.series("u_random_on", "slope")
    assert near(float(off[u - 1]), 1.02, 0.01) and near(ev.mean("u_random_on", "direct_slope"), 0.67, 0.01) and near(float(on[u]), 0.81, 0.01) and near(float(on[u + 5]), 1.01, 0.01)
    assert near(ev.mean("u_youth_on", "direct_slope"), 0.60, 0.01)


def test_the_uplift_jumps_the_level_by_about_137_and_two_thirds_to_a_half_remain():
    u = C.EXP_UPLIFT
    jump = lambda tag: ev.mean(f"u_{tag}_on", "direct_fit") - float(ev.series(f"u_{tag}_off", "fit")[u - 1])
    end = lambda tag: float(ev.series(f"u_{tag}_on", "fit")[-1] - ev.series(f"u_{tag}_off", "fit")[-1])
    assert near(jump("random"), 137, 1.0) and near(jump("youth"), 137, 1.0)
    assert near(end("random"), 104, 1.0) and near(end("youth"), 87, 1.0)


# --- Experiment 7: Anker ---------------------------------------------------------------------------------------------------------------------------

def test_the_anchor_reads_the_scale_level_with_noise():
    got = {k: ev.mean(k, "anchor4") for k, g, _l, _kw in ev.catalog() if g == "anchor"}
    assert near(got["a_closed_2"], 82) and near(got["a_closed_30"], 171) and near(got["a_learns_2"], 71) and near(got["a_learns_30"], 155)
    assert near(got["a_youth_2"], 115) and near(got["a_youth_30"], 204)
    assert max(ev.se(k, "anchor4") for k, g, _l, _kw in ev.catalog() if g == "anchor") < 8.0


# --- Experimente 8 und 9: Teilnahme, Alter, kein Effekt ---------------------------------------------------------------------------------------------

def test_participation_and_ages_shift_the_size_but_not_the_sign():
    assert near(d40("v_youth_base"), 180) and near(d40("v_youth_sched"), 153) and near(d40("v_youth_ages"), 277) and near(d40("v_youth_both"), 250)
    assert near(d40("v_weak_base"), 301) and near(d40("v_weak_sched"), 274) and near(d40("v_weak_ages"), 301) and near(d40("v_weak_both"), 268)
    assert all(d40(k) > 100 for k in ("v_youth_base", "v_youth_sched", "v_youth_ages", "v_youth_both", "v_weak_base", "v_weak_sched", "v_weak_ages", "v_weak_both"))


def test_table_and_rounding_have_no_measurable_effect():
    assert near(d40("n_closed_table"), 154.1, 0.2) and near(d40("n_closed_even"), 153.9, 0.2) and near(d40("n_weak_table"), 300.8, 0.3) and near(d40("n_weak_even"), 301.2, 0.3)
    se = ev.se("n_weak_base", "d40")
    assert abs(d40("n_weak_table") - d40("n_weak_base")) < se and abs(d40("n_weak_even") - d40("n_weak_base")) < 0.1


# --- README -----------------------------------------------------------------------------------------------------------------------------------------

def test_every_test_named_in_the_readme_exists():
    names = set(re.findall(r"`(test_[a-z0-9_]+)`", README))
    assert names, "die README nennt keine Tests"
    existing = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        existing |= set(re.findall(r"^def (test_[a-z0-9_]+)\(", path.read_text(encoding="utf-8"), flags=re.M))
    assert names <= existing, sorted(names - existing)


def test_the_test_count_in_the_readme_is_the_size_of_the_suite():
    stated = [int(m) for m in re.findall(r"(\d+) Tests\b", README)]
    assert stated, "die README nennt keine Testzahl"
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")], capture_output=True, text=True, cwd=ROOT, timeout=300).stdout
    m = re.search(r"(\d+) tests? collected", out)
    assert m, out[-300:]
    assert int(m.group(1)) in stated
