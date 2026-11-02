"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Ausblendbare Regler (fester K, Anhebung, Beitritt des Ankers je nach Schalter) folgen dem KEPT-Muster (DEMO-PLAYBOOK §3):
der Wert eines nicht gezeichneten Reglers wird nie in sein Widget geschrieben, sondern in KEPT abgelegt und unmittelbar vor dem Zeichnen per `seed_widget` eingesetzt. Die
Obergrenzen von Anhebung und Beitritt hängen von den Saisons ab; `clamp_kept` begrenzt vorher."""

import random
from dataclasses import dataclass

import streamlit as st

import dwz_constants as C


def _bool(value) -> bool:
    return str(value).lower() in ("1", "true", "ja")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "seasons_slider": SettingSpec("ss", int, C.DEFAULT_SEASONS, C.SEASONS_MIN, C.SEASONS_MAX),
    "churn_slider": SettingSpec("ch", float, C.DEFAULT_CHURN, C.CHURN_MIN, C.CHURN_MAX),
    "leave_select": SettingSpec("lv", str, C.DEFAULT_LEAVE, choices=C.LEAVE),
    "gap_slider": SettingSpec("gp", int, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "rate_slider": SettingSpec("rt", float, C.DEFAULT_RATE, C.RATE_MIN, C.RATE_MAX),
    "trend_slider": SettingSpec("tr", float, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "kmode_select": SettingSpec("km", str, C.KMODE[0], choices=C.KMODE),
    "kfix_slider": SettingSpec("kf", int, C.DEFAULT_KFIX, C.KFIX_MIN, C.KFIX_MAX),
    "use_a_check": SettingSpec("ua", _bool, True),
    "use_b_check": SettingSpec("ub", _bool, True),
    "use_r_check": SettingSpec("ur", _bool, True),
    "use_floor_check": SettingSpec("uf", _bool, True),
    "special_check": SettingSpec("sw", _bool, True),
    "sched_select": SettingSpec("sc", str, C.SCHED[0], choices=C.SCHED),
    "ages_check": SettingSpec("ag", _bool, False),
    "use_uplift_check": SettingSpec("uu", _bool, False),
    "uplift_slider": SettingSpec("up", int, C.DEFAULT_UPLIFT, C.UPLIFT_MIN, C.SEASONS_MAX),
    "use_anchor_check": SettingSpec("an", _bool, True),
    "join_slider": SettingSpec("j", int, C.DEFAULT_JOIN, C.JOIN_MIN, C.SEASONS_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "seasons": "seasons_slider", "churn": "churn_slider", "leave": "leave_select", "gap": "gap_slider", "rate": "rate_slider", "trend": "trend_slider",
               "kmode": "kmode_select", "kfix": "kfix_slider", "use_a": "use_a_check", "use_b": "use_b_check", "use_r": "use_r_check", "use_floor": "use_floor_check",
               "special": "special_check", "sched": "sched_select", "ages": "ages_check", "use_uplift": "use_uplift_check", "uplift": "uplift_slider", "use_anchor": "use_anchor_check",
               "join": "join_slider", "seed": "seed_input"}
KEPT_KEYS = ("kfix_slider", "uplift_slider", "join_slider")


def _kept() -> dict:
    return st.session_state.setdefault("kept_values", {})


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key in KEPT_KEYS:
            _kept().setdefault(state_key, spec.default)
        elif state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines ausblendbaren Reglers: seinen zuletzt gewählten Wert einsetzen."""
    if state_key not in st.session_state:
        st.session_state[state_key] = _kept().get(state_key, SETTING_SPECS[state_key].default)


def clamp_kept(state_key, lo, hi):
    """Begrenzt den gemerkten (und, falls gezeichnet, den Widget-)Wert auf [lo, hi]; nötig, wenn die Grenzen von anderen Reglern abhängen."""
    kept = _kept()
    kept[state_key] = min(max(kept.get(state_key, SETTING_SPECS[state_key].default), lo), hi)
    if state_key in st.session_state:
        st.session_state[state_key] = min(max(st.session_state[state_key], lo), hi)


def remember(state_key, value):
    _kept()[state_key] = value


def current_value(state_key):
    """Aktueller Wert, auch wenn der Regler in diesem Lauf nicht gezeichnet wird (dann der gemerkte)."""
    if state_key in KEPT_KEYS:
        return st.session_state.get(state_key, _kept().get(state_key, SETTING_SPECS[state_key].default))
    return st.session_state.get(state_key, SETTING_SPECS[state_key].default)


def _store(state_key, value):
    if state_key in KEPT_KEYS:
        _kept()[state_key] = value
        st.session_state.pop(state_key, None)    # beim nächsten Zeichnen aus KEPT einsetzen
    else:
        st.session_state[state_key] = value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        _store(state_key, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            _store(state_key, C.PRESETS[name][key])


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS mit `current_value` (bei ausgeblendeten Reglern der gemerkte Wert). Ein später ergänzter Regler braucht nur seinen
    Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    hidden = {"kfix_slider": lambda: current_value("kmode_select") == C.KMODE[0], "uplift_slider": lambda: not current_value("use_uplift_check"),
              "join_slider": lambda: not current_value("use_anchor_check")}
    for name, preset in C.PRESETS.items():
        same = True
        for state_key, spec in SETTING_SPECS.items():
            if state_key in hidden and hidden[state_key]():
                continue                                  # ein ausgeblendeter Regler bestimmt das Ergebnis nicht
            if current_value(state_key) != preset.get(preset_key_of.get(state_key), spec.default):
                same = False
                break
        if same:
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
                  type="primary" if aktiv == name else "secondary")


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
