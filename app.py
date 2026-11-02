"""DWZ: Gegenmittel gegen Deflation - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zusatzstück der Ranking-Linie hinter Elo-Inflation: Die Deutsche Wertungszahl ist ein Elo-Verfahren mit eingebauten Gegenmitteln gegen Deflation (Erfolgsaufschlag, Bremsfaktor,
Erst-DWZ-Korrektur, Untergrenze, Sonderwertung) und einer einmaligen Anhebung der Skala 2026. Neun Experimente messen in einem simulierten Feld, was diese Regeln erzeugen, wann sie
überschießen und woran man es erkennt. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import dwz_constants as C
import dwz_evaluation as ev
from dwz_layout import dunkles_design, kopf_css
from dwz_pdf_export import build_pdf
from dwz_presets import (
    bounds,
    clamp_kept,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sidebar_presets,
    sync_query_params,
)
from dwz_scenario import Settings, join_max, uplift_max
from dwz_visualization import (
    build_anchor,
    build_balance,
    build_bars,
    build_drift,
    build_drift_lines,
    build_overview,
    build_scale,
    build_slopes,
    build_stack,
)

st.set_page_config(page_title="DWZ – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _rows(keys):
    return ev.rows(list(keys))


def keys_of(group):
    return tuple(k for k, g, _label, _kw in ev.catalog() if g == group)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in dwz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 DWZ: Gegenmittel gegen Deflation")
st.markdown(
    """
Die **Deutsche Wertungszahl (DWZ)** ist ein Elo-Verfahren, das die Skala **nicht sich selbst überlässt**: Wer jung ist oder gut abschneidet, bekommt einen **Erfolgsaufschlag**, schwache Spieler
einen **Bremsfaktor**, Neue eine **Erst-DWZ-Korrektur**, nach unten gilt eine **Untergrenze**, und wer mehr als 300 Punkte über seiner DWZ spielt, wird für seine Gegner höher gewertet
(**Sonderwertung**). 2026 hob der Deutsche Schachbund die Skala außerdem einmalig an. Diese Demo ist ein **Zusatzstück der Ranking-Linie** hinter der Elo-Inflation und misst in einem
simulierten Feld, was diese Gegenmittel erzeugen, wann sie **überschießen** und woran man es erkennt (ein **Anker** konstanter Stärke).

Weiter unten: neun **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Alle Regeln folgen der Wertungsordnung des DSB, gültig ab
8. Juni 2026; die Zahlen sind Simulation und sagen nichts über die reale DWZ.
"""
)

with st.expander("So rechnet die DWZ", expanded=True):
    st.markdown(
        """
1. **Auswertung je Turnier.** Die Gewinnerwartung gegen jeden Gegner folgt einer Normalverteilung (σ = 283 Punkte auf der DWZ-Differenz); ihre Summe ist die Punkterwartung **We**, die erzielten Punkte
   sind **W**. Die neue DWZ ist **R0 + K·(W − We)**, ganzzahlig, nicht unter 1100.
2. **Der Entwicklungsfaktor K** hängt vom **Index** (Zahl der bisherigen Auswertungen), vom **Alter** und von der Wertung ab: Neue und Junge bewegen sich stärker. Dazu kommt der **Erfolgsaufschlag a**
   (nur bei W ≥ We: Jugendliche unter 2000 und Erwachsene unter 1600) und der **Bremsfaktor b** (nur bei W < We unter 1600).
3. **Neue** bekommen ihre **Erst-DWZ** erst, wenn sie mindestens 5 Partien gegen Gewertete gespielt und 0,5 Punkte geholt haben; sie wird gemeinsam iteriert, mit einem Aufschlag unter 2000 und nie unter 1100.
4. **Die Skala:** 2026 wurde sie einmal angehoben (+400 bis 1000, danach 0,64·R + 760 bis 2110). Begründet hat der DSB das mit der FIDE-Anhebung 2024 und einer seit Jahren beobachteten Deflation im unteren Bereich.
5. **Drift** heißt hier: Änderung des Mittels von (Wertung − Stärke) über die Gewerteten, in Punkten. Positiv ist Inflation, negativ Deflation. Die Stärke kennt nur die Simulation.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Feld und Saisons**")
    n = st.slider("Spieler im Feld n", *bounds("n_slider"), step=C.N_STEP, key="n_slider", help="So viele Spieler sind jederzeit im Feld (ohne den Anker).")
    seasons = st.slider("Saisons nach der Einlaufzeit", *bounds("seasons_slider"), key="seasons_slider",
                        help="Acht Saisons vorher lernen die Wertungen die Stärken, ohne Zu- und Abgang. Eine Saison besteht aus zwei Turnieren zu 9 Runden.")
    churn = st.slider("Abgang je Saison (Anteil)", *bounds("churn_slider"), step=C.CHURN_STEP, key="churn_slider", format="%.2f",
                      help="Anteil des Feldes, der nach jeder Saison geht; ebenso viele Neue kommen, ungewertet. 0 = geschlossenes Feld.")
    leave = st.selectbox("Wer geht?", C.LEAVE, key="leave_select", help="Zufällig, oder die Schwächsten nach Wertung (Ungewertete gehen dann zuerst).")
    st.markdown("**Neue und Lernen**")
    gap = st.slider("Neue starten so weit unter ihrer späteren Stärke (Punkte)", *bounds("gap_slider"), step=C.GAP_STEP, key="gap_slider",
                    help="0 = Neue kommen mit ihrer Stärke. Sonst sind sie Jugendliche (Alter 14) und holen je Saison einen Teil des Abstands auf („lernende Jugend“).")
    rate = st.slider("Aufholen je Saison (Anteil des Abstands)", *bounds("rate_slider"), step=C.RATE_STEP, key="rate_slider", format="%.2f", help="Wirkt nur, wenn die Neuen unter ihrer Stärke starten.")
    trend = st.slider("Das ganze Feld wird je Saison besser (+) oder schlechter (−), Punkte", *bounds("trend_slider"), step=C.TREND_STEP, key="trend_slider", format="%.1f",
                      help="Änderung der Stärke aller Spieler je Saison; Neue kommen auf dem Niveau des Feldes. Der Anker bleibt, wo er ist.")
    st.markdown("**Regeln der Wertungsordnung**")
    kmode = st.selectbox("Entwicklungsfaktor K", C.KMODE, key="kmode_select", help="Nach der WO aus Index, Alter und Wertung, oder ein fester Wert für alle (ohne Aufschlag und Bremse).")
    if kmode == C.KMODE[1]:
        seed_widget("kfix_slider")
        kfix = st.slider("Fester K", *bounds("kfix_slider"), key="kfix_slider", help="K für alle Spieler; Aufschlag a und Bremsfaktor b entfallen.")
        remember("kfix_slider", kfix)
    use_a = st.checkbox("Erfolgsaufschlag a", key="use_a_check", help="Nur bei W ≥ We und nur im DWZ-K: Jugendliche bis 20 unter 2000 bekommen (2000 − R0)/30, Ältere unter 1600 bekommen 4.")
    use_b = st.checkbox("Bremsfaktor b", key="use_b_check", help="Nur bei W < We unter 1600 und nur im DWZ-K: K wird mit (R0 + 800)/2400 multipliziert, schwache Spieler verlieren weniger.")
    use_r = st.checkbox("Erst-DWZ-Korrektur r = 0,08", key="use_r_check", help="Eine Erst-DWZ unter 2000 wird um 8 % des Abstands zu 2000 angehoben.")
    use_floor = st.checkbox("Untergrenze 1100", key="use_floor_check", help="Keine DWZ fällt unter 1100, die Differenz wird gutgeschrieben (und ist die kleinste Erst-DWZ).")
    special = st.checkbox("Sonderwertung 300 Punkte", key="special_check", help="Liegt die Turnierleistung eines Spielers mehr als 300 über seiner DWZ, rechnen seine Gegner mit dieser Leistung.")
    sched = st.selectbox("Turniere", C.SCHED, key="sched_select", help="Alle spielen zwei Turniere je Saison, oder Aktivität nach Spieler plus ein Mannschaftsturnier zu 7 Runden (WO 4.2.4.3).")
    ages = st.checkbox("Altersverteilung", key="ages_check", help="Statt fester Alter (Gründer 35, Neue 30 bzw. 14): Gründer zu einem Viertel jung, Neue 18 bis 60 bzw. 8 bis 17 Jahre; die Jugend lernt nur bis 20.")
    st.markdown("**Anhebung 2026**")
    use_uplift = st.checkbox("Einmalige Anhebung der Skala", key="use_uplift_check", help="Anhang 4 der WO: +400 bis 1000, 0,64·R + 760 bis 2110, darüber unverändert.")
    if use_uplift:
        clamp_kept("uplift_slider", C.UPLIFT_MIN, uplift_max(int(seasons)))
        seed_widget("uplift_slider")
        uplift = st.slider("Anhebung zu Beginn von Saison (nach der Einlaufzeit)", C.UPLIFT_MIN, uplift_max(int(seasons)), key="uplift_slider", help="Saisons, die vor der Anhebung gespielt sind.")
        remember("uplift_slider", uplift)
    st.markdown("**Referenzspieler (Anker)**")
    use_anchor = st.checkbox("Anker mitspielen lassen", key="use_anchor_check", help="Ein Spieler mit konstanter Stärke (1500 per Konvention), der nie geht und wie die anderen gewertet wird.")
    if use_anchor:
        clamp_kept("join_slider", C.JOIN_MIN, join_max(int(seasons)))
        seed_widget("join_slider")
        join = st.slider("Anker tritt bei zu Beginn von Saison", C.JOIN_MIN, join_max(int(seasons)), key="join_slider", help="Saisons, die vor seinem Beitritt gespielt sind; er startet bei 1500.")
        remember("join_slider", join)
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

kfix_v = int(current_value("kfix_slider"))
uplift_v = int(current_value("uplift_slider"))
join_v = int(current_value("join_slider"))
sync_query_params({"n_slider": int(n), "seasons_slider": int(seasons), "churn_slider": round(float(churn), 2), "leave_select": leave, "gap_slider": int(gap),
                   "rate_slider": round(float(rate), 2), "trend_slider": round(float(trend), 1), "kmode_select": kmode, "kfix_slider": kfix_v, "use_a_check": bool(use_a),
                   "use_b_check": bool(use_b), "use_r_check": bool(use_r), "use_floor_check": bool(use_floor), "special_check": bool(special), "sched_select": sched,
                   "ages_check": bool(ages), "use_uplift_check": bool(use_uplift), "uplift_slider": uplift_v, "use_anchor_check": bool(use_anchor), "join_slider": join_v,
                   "seed_input": int(seed)})

settings = Settings(int(n), int(seasons), round(float(churn), 2), leave, int(gap), round(float(rate), 2), round(float(trend), 1), kmode, kfix_v, bool(use_a), bool(use_b), bool(use_r),
                    bool(use_floor), bool(special), sched, bool(ages), bool(use_uplift), uplift_v, bool(use_anchor), join_v, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Skala im gezeigten Lauf")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Drift nach {settings.seasons} Saisons (Punkte)", ev.signed(a.drift[-1]), delta=f"Fähigkeit {ev.signed(a.level[-1])}, Wertung {ev.signed(a.rating[-1])}", delta_color="off",
          help=f"Änderung von (Wertung − Stärke) über die Gewerteten seit dem Ende der Einlaufzeit. Ein einzelner Lauf streut um rund ±{ev.de(C.NOISE)} Punkte.")
m2.metric("Steigung der Wertung auf die Stärke", ev.de(float(a.slope[-1]), 2), delta=f"am Ende der Einlaufzeit {ev.de(a.baseline[0], 2)}", delta_color="off",
          help="1 heißt: 100 Punkte mehr Stärke sind 100 Punkte mehr Wertung. Unter 1 ist die Skala gestaucht, über 1 gedehnt.")
anchor_seen = [i for i, v in enumerate(a.anchor) if v == v]
if anchor_seen:
    last = anchor_seen[-1]
    m3.metric("Anker am Ende", ev.de(float(a.anchor[last])), delta=f"{ev.signed(float(a.anchor[last]) - C.BASE)} gegen seine Stärke 1500", delta_color="off",
              help="Ein Spieler konstanter Stärke (1500 per Konvention), der selbst nach denselben Regeln gewertet wird; er liest die Skalenlage ab, mit Rauschen.")
else:
    m3.metric("Anker am Ende", "–", delta="kein Anker", delta_color="off")
extra = a.bal["a"] + a.bal["b"] + a.bal["special"] + a.bal["floor"]
m4.metric("Punkte aus den Regeln", ev.signed(extra), delta=f"{ev.signed(extra / settings.n, 1)} je Spieler", delta_color="off",
          help="Punkte, die Erfolgsaufschlag, Bremsfaktor, Sonderwertung und Untergrenze zur Summe der Wertungen hinzugefügt haben; bei festem K ohne diese Regeln null.")

st.plotly_chart(build_overview(a), width="stretch", key="overview_chart")
st.caption("Wertungsmittel der Gewerteten und ihre mittlere Fähigkeit nach jeder Saison (DWZ-Skala); die Lücke zwischen beiden ist die Drift. Der Anker (violett) ist ein Spieler konstanter Stärke "
           "und rauscht. Bei der Anhebung springt die Wertung, die Fähigkeit nicht.")
st.plotly_chart(build_drift(a), width="stretch", key="drift_chart")
st.caption("Drift je Saison (Wertung minus Fähigkeit).")
st.plotly_chart(build_scale(a), width="stretch", key="scale_chart")
st.caption("Steigung der Wertung auf die wahre Stärke: Die Anhebung staucht sie, die feste Gewinnerwartung dehnt sie zurück.")
st.plotly_chart(build_balance(a), width="stretch", key="balance_chart")
st.caption("Bilanz der Wertungssumme über die gemessenen Saisons: Sie ändert sich nur durch Eintritt (Erst-DWZ der Neuen) minus Abgang, die Punkte aus Erfolgsaufschlag, Bremsfaktor und Sonderwertung, "
           "ungleiches K0 mit Rundung und Untergrenze, die Anhebung und den Beitritt des Ankers.")

st.dataframe(pd.DataFrame(ev.field_table(a)), hide_index=True, width="stretch")
st.download_button("📄 PDF des aktuellen Laufs", data=build_pdf(a), file_name="dwz.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wo steht die Skala?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

EXP = (f"{C.EXP_N} Spieler (feste Instanz, unabhängig vom Regler), {C.EXP_SEASONS} Saisons nach {C.BURN_IN} Saisons Einlaufzeit, {ev.pct(C.EXP_CHURN)} % Abgang je Saison, Mittel über 60 feste Läufe "
       "(vorberechnet, `tools/build_experiments.py`).")

st.subheader("🔬 Experiment 1: Was die Gegenmittel erzeugen")
st.caption(f"Geschlossenes Feld: niemand geht, niemand kommt, alle Stärken sind fest. {EXP} Jede Zeile fügt dem Grundfall „nur K aus Index, Alter, Wertung“ eine Regel hinzu.")
rows1 = _rows(keys_of("closed"))
st.plotly_chart(build_bars(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_remedies(_rows(keys_of("closed") + keys_of("scen"))))
st.dataframe(pd.DataFrame([{"Konfiguration": r["label"], "Drift nach 40 Saisons": round(r["d40"]), "± Standardfehler": round(r["se"], 1), "Punkte aus a": round(r["a"]), "aus b": round(r["b"]),
                            "aus Sonderwertung": round(r["special"]), "aus Untergrenze": round(r["floor"])} for r in rows1]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Lernende Jugend – Deflation und Überschuss")
st.caption(f"Neue kommen {C.DEFAULT_GAP} Punkte unter ihrer späteren Stärke (Alter 14) und holen 12 % des Abstands je Saison auf; sonst wie Experiment 1.")
rows2 = _rows(keys_of("youth"))
st.plotly_chart(build_bars(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_youth(rows2))
st.plotly_chart(build_drift_lines(rows2, keys=("y_none", "y_bare", "y_a", "y_all")), width="stretch", key="exp2_lines")
st.caption("Verlauf der mittleren Drift: ohne Gegenmittel, nur K aus Index und Alter, mit Erfolgsaufschlag und mit allen Regeln.")

st.subheader("🔬 Experiment 3: Wer erzeugt Punkte? Die Bilanz")
st.caption(f"{EXP} Gezeigt sind die Posten der Bilanz der Wertungssumme, je Feldspieler: Eintritt minus Abgang, Erfolgsaufschlag, Bremsfaktor, Sonderwertung, der Rest (ungleiches K0, Rundung, Untergrenze).")
scen_keys = tuple(k for k in keys_of("scen"))
rows3 = _rows(scen_keys)
st.plotly_chart(build_stack(rows3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_balance(rows3))
st.dataframe(pd.DataFrame([{"Szenario": r["label"], "Drift nach 40 Saisons": round(r["d40"]), "Erst-DWZ der Neuen − Stärke": round(r["entry_err"]), "Wertung der Gehenden − Stärke": round(r["leave_err"]),
                            "Punkte im Spiel je Spieler": round(ev.created(r) / C.EXP_N, 1)} for r in rows3]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 4: Die Sonderwertung 300")
st.caption(f"{EXP} Verglichen werden der Grundfall und die Sonderwertung allein im geschlossenen Feld sowie das lernende Feld ohne und mit allen Regeln.")
rows4 = _rows(("c_bare", "c_special", "c_all", "s_learns_none", "s_learns_all"))
st.plotly_chart(build_bars(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_special(_rows(keys_of("closed") + keys_of("scen"))))

st.subheader("🔬 Experiment 5: Die Eintrittsregel der Erst-DWZ")
st.caption(f"{EXP} Ohne Gegenmittel (fester K 28, keine Zuschläge) und mit allen Regeln, bei zufälligem Abgang und wenn die Schwächsten gehen.")
rows5 = _rows(("s_random_none", "s_weak_none", "s_random_all", "s_weak_all"))
st.plotly_chart(build_bars(rows5, field="entry_err", title="Erst-DWZ der Neuen minus ihre Stärke (Punkte)"), width="stretch", key="exp5_chart")
verdict(*ev.verdict_entry(_rows(keys_of("scen"))))
st.dataframe(pd.DataFrame([{"Szenario": r["label"], "Erst-DWZ der Neuen − Stärke": round(r["entry_err"]), "Wertung der Gehenden − Stärke": round(r["leave_err"]),
                            "Drift nach 40 Saisons": round(r["d40"])} for r in rows5]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 6: Die Anhebung 2026 und die Rückdehnung")
st.caption(f"{EXP} Die Anhebung gilt zu Beginn von Saison {C.EXP_UPLIFT + 1} nach der Einlaufzeit. Gezeigt ist die Steigung der Wertung auf die Stärke.")
rows6 = _rows(keys_of("uplift"))
st.plotly_chart(build_slopes(keys_of("uplift")), width="stretch", key="exp6_chart")
verdict(*ev.verdict_uplift(rows6))
st.dataframe(pd.DataFrame([{"Konfiguration": r["label"], "Drift nach 40 Saisons": round(r["d40"]), "Skalenlage bei Stärke 1500 (Änderung)": round(r["fit"]), "Steigung am Ende": ev.de(r["slope"], 2)}
                           for r in rows6]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 7: Der Anker über die Zeit")
st.caption(f"{EXP} Ein Spieler konstanter Stärke (1500 per Konvention) tritt nach 2, 15 oder 30 Saisons bei; gezeigt ist seine Wertung vier Saisons später, Fehlerbalken: Standardfehler.")
rows7 = _rows(keys_of("anchor"))
st.plotly_chart(build_anchor(rows7), width="stretch", key="exp7_chart")
verdict(*ev.verdict_anchor(rows7))

st.subheader("🔬 Experiment 8: Teilnahme und Altersverteilung")
st.caption(f"{EXP} Grundfall gegen „Aktivität nach Spieler plus Mannschaftsturnier“, gegen die Altersverteilung und gegen beides.")
rows8 = _rows(keys_of("variants"))
st.plotly_chart(build_bars(rows8), width="stretch", key="exp8_chart")
verdict(*ev.verdict_variants(rows8))

st.subheader("🔬 Experiment 9: Was keinen Effekt hat")
st.caption(f"{EXP} Die gedruckte Tabelle P(D) der WO (D ganzzahlig, p auf drei Stellen) und das numpy-Rundungsverfahren gegen den Grundfall.")
rows9 = _rows(keys_of("noeffect"))
st.plotly_chart(build_bars(rows9), width="stretch", key="exp9_chart")
verdict(*ev.verdict_noeffect(rows9))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die absolute Fähigkeit ist definiert** | Im Modell gehen nur Differenzen in die Gewinnchance ein; der Nullpunkt (1500) ist eine Konvention. „Inflation“ ist nur gegen einen Anker oder eine externe Messlatte eine Aussage. | Elo-Inflation |
| **Das Spielmodell ist die Wertung** | Die Partien folgen derselben Normalverteilung, mit der die DWZ rechnet (σ = 283); echte Partien weichen davon ab, und der Remisanteil ist ein fester Wert (30 % bei Gleichstand). | – |
| **Zu- und Abgang sind stilisiert** | Feste Abgangsquote, feste Verteilungen, keine Rückkehrer, keine Vereinswechsel, keine Vorhaltefrist von 5 Jahren, keine FIDE-Elo als erste DWZ, keine Zweikämpfe, keine Hybridturniere. | – |
| **Die Erst-DWZ ist vereinfacht** | Zusatzstufen, Restpartien (Verfall nach 2 Jahren) und Gesamtiteration sind nach der WO modelliert; Mannschaftskämpfe sind ein Turnier zu 7 Runden, keine Brettpaarungen. | – |
| **Die Jugend ist gewählt** | Anteil, Abstand und Tempo der Lernenden sind frei; die reale Jugendbewegung ist eine andere. Der Überschuss der Gegenmittel ist eine Aussage über die Mechanik, nicht über die DWZ. | – |
| **Die Anhebung ist modelliert, nicht bewertet** | Das Modell zeigt Stauchung, Rückdehnung und den bleibenden Niveausprung; ob 0,64 sachgerecht ist, kann es nicht sagen. | – |
| **Der Anker ist genau** | Seine Wertung rauscht um etwa ±40 Punkte je Lauf und unterliegt selbst den Regeln (a, b, Sonderwertung); Verschiebungen unter etwa 20 Punkten sind nicht zu deuten. | Elo-Inflation |
| **Ein Verfahren** | Nur die DWZ nach der WO ab dem 8. Juni 2026; die frühere Formel mit 800/(E + n) ist nicht modelliert. Wie andere Verfahren über die Zeit driften, vergleicht ein eigenes Stück. | Welches Verfahren, gemessen woran? |
| **Alles ist simuliert** | Wahre Stärken aus dem Generator, danach Abgang, Zugang und Lernen; die Zahlen sind Simulation, keine Aussage über Schach oder einen anderen Sport. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gewinnerwartung.** Für Spieler 1 gegen 2 mit $D = R_1 - R_2$: $P(D) = \Phi(D/\sigma)$, $\sigma = 200\sqrt{2} \approx 282{,}8$ (WO 3.3.1). Die Punkterwartung eines Turniers ist $W_e = \sum P(D_i)$,
die erzielten Punkte sind $W$ (Sieg 1, Remis ½).

**Folge-DWZ.** $R_n = R_0 + K\,(W - W_e)$, ganzzahlig, $R_n \ge 1100$ (WO 3.5.2). $K = K_0\,b + a$, auf eine Dezimale gerundet, $K \le 80$. Der Grundwert $K_0$ steht in einer Tabelle nach Index (1 bis 10,
größer 10) und Gruppe (bis 25 Jahre unter 2200 · ab 26 Jahren unter 2000 · darüber), von 60 bei Index 1 und 2 bis 25 bei Index über 10.
Erfolgsaufschlag $a$ nur bei $W \ge W_e$: $(2000 - R_0)/30$ für Jugendliche bis 20 Jahre unter 2000, $4$ für Ältere unter 1600. Bremsfaktor $b = (R_0 + 800)/2400$ nur bei $W < W_e$ und $R_0 < 1600$.

**Erst-DWZ.** Ein Ungewerteter mit mindestens 5 Partien gegen Gewertete (1. Zusatzstufe) oder gegen die Auswahl I (2. Zusatzstufe) und mindestens 0,5 Punkten bekommt aus der Gesamtiteration
$R_i$ mit $p' = \tfrac12 + (W - W_e)/n$, $R_i \leftarrow R_i + D(p')$ bis $\max |D'| < 0{,}1$; bei 100 % zählt ein fiktives Remis gegen den Gegnerschnitt. Die Erst-DWZ ist $R_1 = R_i$ ab 2000, sonst
$R_i + r\,(2000 - R_i)$ mit $r = 0{,}08$, unter 1100 $1100 + 9\,r\,R_i/11$. Partien gegen die Auswahl I, die nicht zählen, werden zwei Jahre als Restpartien gesammelt.

**Sonderwertung.** Liegt die Turnierleistung $R_h$ ($W_e(R_h) = W$) eines Etablierten mit mindestens 5 Partien mehr als 300 über $R_0$, rechnen seine etablierten Gegner mit $R_h$ statt $R_0$.
Die Summe der Wertungsänderungen ist dann $\sum_j K_j \bigl[P(R_j - R_{0,X}) - P(R_j - R_{h,X})\bigr]$ größer null, weil der Ausnahmespieler $X$ selbst unverändert gewertet wird.

**Anhebung.** $R \mapsto R + 400$ für $R \le 1000$, $0{,}64\,R + 760$ für $1000 < R \le 2110$, sonst unverändert (WO Anhang 4).

**Bilanz.** Die Summe $\Sigma$ der Wertungen ändert sich um Erst-DWZ der Neuen − Wertungen der Gehenden + Punkte aus $a$ + aus $b$ + aus der Sonderwertung + Rest (ungleiches $K_0$) + Rundung + Untergrenze + Anhebung
+ Beitritt des Ankers; die Posten sind exakt (Orakel im Repo). **Drift.** $D_t = \overline{R - s}_t - \overline{R - s}_0$ über die Gewerteten, mit der wahren Stärke $s$.

**Zufall.** Normalverteilung per Box-Muller aus der Gleichverteilten des Generators (SplitMix64); Paarung durch Sortieren zufälliger Schlüssel je Runde unter den Teilnehmern; alle Zufallszahlen eines Laufs
stehen in fester Reihenfolge fest, unabhängig von den Schaltern der Regeln.

**Literatur.**
- Deutscher Schachbund: *Wertungsordnung*, gültig ab 8. Juni 2026 (Online-Fassung und PDF `WO_2026.pdf`, eingesehen am 2026-11-02); *Die neue DWZ-Abfrage ist online und die DWZ ist angehoben*, Meldung vom 28. Juli 2026.
- Elo, A. E. (1978): *The Rating of Chessplayers, Past and Present.* Arco (das Grundverfahren nach der üblichen Darstellung; das Buch wurde nicht eingesehen).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
- Empirische Befunde zur Inflation und Deflation der DWZ wurden nicht herangezogen; die Zahlen dieser Demo sind Simulation.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
