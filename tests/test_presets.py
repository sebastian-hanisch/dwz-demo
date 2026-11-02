"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Adressparameter; feste Instanzen der Experimente."""

import dwz_constants as C
import dwz_presets as P
from dwz_scenario import Settings

PRESET_FIELDS = {"n": "n", "seasons": "seasons", "churn": "churn", "leave": "leave", "gap": "gap", "rate": "rate", "trend": "trend", "kmode": "kmode", "kfix": "kfix", "use_a": "use_a",
                 "use_b": "use_b", "use_r": "use_r", "use_floor": "use_floor", "special": "special", "sched": "sched", "ages": "ages", "use_uplift": "use_uplift", "uplift": "uplift",
                 "use_anchor": "use_anchor", "join": "join", "seed": "seed"}


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    p = C.PRESETS[first]
    assert Settings(**{field: p[key] for key, field in PRESET_FIELDS.items()}) == Settings()


def test_preset_keys_are_exactly_the_settings_fields():
    assert set(PRESET_FIELDS.values()) == set(Settings.__dataclass_fields__)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_parameters_are_unique():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_hidden_widgets_are_exactly_the_switch_dependent_ones():
    assert set(P.KEPT_KEYS) == {"kfix_slider", "uplift_slider", "join_slider"}


def test_each_preset_shows_its_mechanism():
    p = C.PRESETS
    assert p["Geschlossenes Feld"]["churn"] == 0.0
    assert p["Das Feld lernt"]["trend"] > 0 and p["Die Schwächsten gehen"]["leave"] == C.LEAVE[1]
    assert p["Anhebung 2026"]["use_uplift"] and p["Lernende Jugend, DWZ-Regeln"]["gap"] > 0
    ohne = p["Lernende Jugend, ohne Gegenmittel"]
    assert ohne["kmode"] == C.KMODE[1] and not any(ohne[k] for k in ("use_a", "use_b", "use_r", "use_floor", "special"))
    mit = p["Lernende Jugend, DWZ-Regeln"]
    assert mit["kmode"] == C.KMODE[0] and all(mit[k] for k in ("use_a", "use_b", "use_r", "use_floor", "special"))


def test_bool_caster_reads_the_address_values():
    assert P._bool("True") and P._bool("1") and P._bool("true") and not P._bool("False") and not P._bool("0") and not P._bool("")


def test_sweep_instances_are_fixed_and_distinct():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60


def test_experiment_marks_fit_into_the_experiment_instance():
    assert max(C.EXP_MARKS) == C.EXP_SEASONS and C.EXP_UPLIFT + 5 < C.EXP_SEASONS and max(C.ANCHOR_ERAS) + C.ANCHOR_STAY <= C.EXP_SEASONS
