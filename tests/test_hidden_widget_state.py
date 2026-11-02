"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück. Die Obergrenzen von Anhebung und Beitritt hängen von den Saisons ab."""

import os

from streamlit.testing.v1 import AppTest

import dwz_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 600
HIDDEN = {"kfix_slider", "uplift_slider", "join_slider"}


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def widget_keys(at):
    return {s.key for s in at.slider}


def test_widgets_are_only_drawn_when_their_switch_is_on():
    at = fresh()
    assert widget_keys(at) & HIDDEN == {"join_slider"}              # der Anker ist voreingestellt
    at.selectbox(key="kmode_select").set_value(C.KMODE[1]).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == {"join_slider", "kfix_slider"}
    at.checkbox(key="use_uplift_check").set_value(True).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == HIDDEN
    at.checkbox(key="use_anchor_check").set_value(False).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == HIDDEN - {"join_slider"}
    assert not at.exception


def test_hidden_values_survive_and_are_not_written_into_hidden_widgets():
    at = fresh()
    at.selectbox(key="kmode_select").set_value(C.KMODE[1]).run(timeout=TIMEOUT)
    at.slider(key="kfix_slider").set_value(40).run(timeout=TIMEOUT)
    at.selectbox(key="kmode_select").set_value(C.KMODE[0]).run(timeout=TIMEOUT)
    assert "kfix_slider" not in at.session_state
    at.selectbox(key="kmode_select").set_value(C.KMODE[1]).run(timeout=TIMEOUT)
    assert at.slider(key="kfix_slider").value == 40
    assert not at.exception


def test_join_and_uplift_follow_a_shrinking_number_of_seasons():
    at = fresh()
    at.checkbox(key="use_uplift_check").set_value(True).run(timeout=TIMEOUT)
    at.slider(key="join_slider").set_value(35).run(timeout=TIMEOUT)
    at.slider(key="uplift_slider").set_value(38).run(timeout=TIMEOUT)
    at.slider(key="seasons_slider").set_value(C.SEASONS_MIN).run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="join_slider").value <= C.SEASONS_MIN - 2 and at.slider(key="uplift_slider").value <= C.SEASONS_MIN - 1


def test_preset_sets_values_and_hidden_widgets_for_the_next_drawing():
    at = fresh()
    at.sidebar.button(key="preset_Das Feld lernt").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="leave_select").value == C.LEAVE[0] and abs(at.slider(key="trend_slider").value - 3.5) < 1e-9 and at.slider(key="gap_slider").value == 0
    at.sidebar.button(key="preset_Geschlossenes Feld").click().run(timeout=TIMEOUT)
    assert at.slider(key="churn_slider").value == 0.0 and abs(at.slider(key="trend_slider").value) < 1e-9
    at.sidebar.button(key="preset_Anhebung 2026").click().run(timeout=TIMEOUT)
    assert at.checkbox(key="use_uplift_check").value and at.slider(key="uplift_slider").value == C.DEFAULT_UPLIFT
    at.sidebar.button(key="preset_Lernende Jugend, ohne Gegenmittel").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="kmode_select").value == C.KMODE[1] and not at.checkbox(key="special_check").value and at.slider(key="kfix_slider").value == C.DEFAULT_KFIX
    assert not at.exception


def test_the_anchor_switch_removes_the_anchor_from_the_table():
    at = fresh()
    assert any(r["Spieler"] == "Anker" for r in at.dataframe[0].value.to_dict("records"))
    at.checkbox(key="use_anchor_check").set_value(False).run(timeout=TIMEOUT)
    assert not any(r["Spieler"] == "Anker" for r in at.dataframe[0].value.to_dict("records"))
    assert not at.exception
