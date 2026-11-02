# DWZ: Gegenmittel gegen Deflation – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-dwz-demo.streamlit.app/)** -->

Zusatzstück **7c** der **Ranking-Linie** (Konzepte-Demo, Methodenreihe), Fortsetzung von `elo-inflation-demo` (7b): Die **Deutsche Wertungszahl (DWZ)** ist ein Elo-Verfahren, das die Skala nicht sich selbst
überlässt. Sie hat **Gegenmittel gegen Deflation** (Erfolgsaufschlag a, Bremsfaktor b, Erst-DWZ-Korrektur r, Untergrenze 1100, Sonderwertung bei mehr als 300 Punkten über der DWZ) und seit 2026 eine
**einmalige Anhebung der Skala**. Die Demo rechnet nach der **Wertungsordnung des Deutschen Schachbundes, gültig ab 8. Juni 2026**, in einem **simulierten, offenen Feld** (neutrales Turnier, wahre Stärken
bekannt, Abgang und Zugang, lernende Jugend, ein Anker konstanter Stärke) und misst, was diese Regeln erzeugen, wann sie überschießen und woran man es erkennt. Alles ist simuliert, es gibt keinen echten
Datensatz; die Gründerstärken und der Zufallsstrom kommen aus dem Generator der Linie (Version 1, wortgleich).

## Kernfrage

Wohin bewegt sich die DWZ-Skala, wenn Neue ungewertet kommen, lernen und andere gehen, und was ändern die eingebauten Korrekturen daran? **Die Gegenmittel erzeugen Punkte.** Gemessen (60 feste Läufe, 60 Spieler,
40 Saisons nach 8 Saisons Einlaufzeit, 15 % Abgang je Saison): Kommen die Neuen 350 Punkte unter ihrer späteren Stärke und lernen, **fällt** die Skala ohne Gegenmittel um **130** Punkte (Deflation); mit allen Regeln der
Wertungsordnung **steigt** sie um **180**. Schon im **geschlossenen Feld** ohne Zu- und Abgang steigt sie mit allen Regeln um **154** Punkte, und die **Sonderwertung 300** dreht das Vorzeichen, wenn das ganze
Feld lernt (−115 → +163). Die Anhebung 2026 staucht die Steigung der Wertung auf die Stärke auf 0,67 und dehnt sie in fünf Saisons auf 1,01 zurück, der Niveausprung (+137) bleibt zu zwei Dritteln bis zur
Hälfte. **Ob die reale DWZ so reagiert, zeigt die Simulation nicht.**

## Befunde und Korrekturen gegenüber dem Plan

- **Die aktuelle Formel ist nicht `800/(E + n)`.** Das ist die Fassung vor 2026 (Sekundärquellen); die Wertungsordnung ab 8. Juni 2026 rechnet `Rn = R0 + K·(W − We)` mit K aus Index, Alter und Wertung
  (Anhang 1.3). Die Demo bildet nur die Fassung 2026 ab.
- **Die Plan-Zahlen verschieben sich.** Die Vorab-Messreihe (Python-Zufallsstrom, Teilnahme, Alter und Tabelle zugeschaltet) hatte „lernende Jugend stark, DWZ-Regeln“ bei +252 und „ohne Gegenmittel“ bei −84;
  die Demo (SplitMix64, Grundfall) misst +180 und −130. Die Vorzeichen und die Reihenfolge der Bausteine bleiben, die Größen ändern sich um 5 bis 30 % (Experiment 8), die Erwartung aus dem Plan hielt.
- **Der Drift von „Die Schwächsten gehen“ (+301) ist erklärt, nicht mehr offen:** Er folgt aus der Hürde der Erst-DWZ (5 Partien und 0,5 Punkte), die aus dem Schätzer eine Auswahl der Glücklichen macht, der Untergrenze
  1100 und dem fehlenden Anker (Experiment 3 und 5; Kontrollversuch in `test_the_threshold_of_half_a_point_selects_the_lucky_newcomers`). Meine frühere Erwartung („im Gleichgewicht Drift ≈ 0“) war falsch: Es gibt keine rückstellende Kraft.
- **Neu gegenüber dem Plan:** die Sonderwertung als eigenes, großes Experiment (Experiment 4; im Plan als „nicht modelliert“ geführt), die Buchhaltung der Wertungssumme mit einem exakten Orakel, der Abgleich mit dem
  durchgerechneten Beispielturnier der Wertungsordnung (Anhang 3.1) und die Zeile „Was keinen Effekt hat“.

## Modell

- **Feld:** `n` Spieler (wahre Stärke N(1500; 200), Startwertung 1500, Index > 10, Alter 35) und ein Platz für den Anker (Stärke 1500, Index > 10). Eine Saison besteht aus zwei Turnieren zu 9 Runden, an denen alle
  teilnehmen (zufällige Paarung, bei ungerader Zahl setzt einer aus), oder aus zwei Turnieren nach Aktivität der Spieler plus einem Mannschaftsturnier zu 7 Runden (WO 4.2.4.3). Partien folgen derselben Normalverteilung,
  mit der die DWZ rechnet (σ = 200·√2); der Remisanteil ist `0,3·4p(1−p)`, die Punkterwartung bleibt p.
- **Auswertung je Turnier (WO 3.5.2):** `Rn = R0 + K·(W − We)`, ganzzahlig (kaufmännisch), nicht unter 1100. `K = K0·b + a`, auf eine Dezimale gerundet, höchstens 80. `K0` aus der Tabelle nach Index (1 bis 10, > 10) und Gruppe,
  `a = (2000 − R0)/30` für Jugendliche bis 20 Jahre unter 2000 und `4` für Ältere unter 1600 (nur bei W ≥ We), `b = (R0 + 800)/2400` bei W < We unter 1600. Der Index steigt um 1 je Auswertung (WO 2.2.4).
- **Erst-DWZ (WO 3.4.1 bis 3.4.4, Anhang 1.1):** Ungewertete bekommen sie, wenn sie mindestens 5 Partien gegen Gewertete (1. Zusatzstufe) bzw. gegen die Auswahl I (2. Zusatzstufe) und mindestens 0,5 Punkte haben; Restpartien
  verfallen nach 2 Jahren; die Gesamtiteration `p' = 0,5 + (W − We)/n`, `D' = D(p')` läuft bis max |D'| < 0,1, bei 100 % mit fiktivem Remis gegen den Gegnerschnitt; `R1 = Ri + 0,08·(2000 − Ri)` unter 2000.
- **Sonderwertung (WO 3.6.3):** Liegt die Turnierleistung Rh (≥ 5 Partien) mehr als 300 über R0, rechnen die etablierten Gegner mit Rh.
- **Abgang und Zugang:** nach der Einlaufzeit gehen `churn·n` Spieler (zufällig, oder die Schwächsten; Ungewertete zuerst); ebenso viele Neue kommen **ungewertet**. Mit einem Abstand `gap` > 0 sind sie Jugendliche
  (Alter 14) und holen je Saison einen Anteil des Abstands zu ihrer Spitzenstärke auf; ein Trend verändert die Stärke aller.
- **Anhebung (Anhang 4):** `+400` bis 1000, `0,64·R + 760` bis 2110, darüber unverändert.
- **Maße:** *Drift* = Änderung von mean(Wertung − Stärke) über die Gewerteten, gegenüber dem Ende der Einlaufzeit (Punkte; positiv Inflation, negativ Deflation); *Steigung* der Wertung auf die Stärke; *Skalenlage* =
  angepasste Wertung bei Stärke 1500; *Bilanz* der Wertungssumme (Eintritt, Abgang, a, b, Sonderwertung, ungleiches K0 mit Rundung, Untergrenze, Anhebung, Anker).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanz mit 60 Spielern, 40 Saisons nach 8 Saisons Einlaufzeit, 15 % Abgang,
  unabhängig vom Regler. Die 54 Konfigurationen (je 60 Läufe zu etwa 0,2 s) sind **vorberechnet** in `data/experiments.json` (`tools/build_experiments.py`); `tests/test_experiments_data.py` rechnet je
  Konfiguration einen Lauf frisch nach und vergleicht ihn mit der Datei.
- **Gepaarte Vergleiche:** Alle Zufallszahlen eines Laufs stehen in fester Reihenfolge und Größe fest, unabhängig von den Schaltern der Regeln; Konfigurationen eines Experiments teilen sich dieselben Gründer, Paarungen
  und Ausgänge, soweit die Regeln es zulassen.
- **Bilanz:** Die Posten der Wertungssumme addieren sich exakt (Rest unter 1e-8 im Orakel); „Spiel“ zahlt nur aus, was Regeln vorsehen.
- **Abgleich mit der Wertungsordnung:** Das durchgerechnete Beispielturnier (Anhang 3.1, 16 Spieler, Gesamtiteration, Sonderwertung) wird in `tests/test_literatur_dwz.py` Zahl für Zahl nachgerechnet.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie); Normalverteilung per Box-Muller; Beta(2, 2) der Aktivität als Median dreier Gleichverteilter.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 60 Spieler, 40 Saisons nach der Einlaufzeit, 15 % Abgang, Drift in Punkten (Mittel), wenn nicht anders genannt; der Fehler jedes Mittelwerts liegt bei 1 bis 7 Punkten.

| Frage | Befund | Test |
|---|---|---|
| Geschlossenes Feld ohne Gegenmittel / nur K aus Index und Alter | **±0** / ±0: Die Skala steht still. | `test_a_closed_field_without_remedies_and_with_only_k_from_index_and_age_stands_still` |
| Geschlossenes Feld mit allen Regeln | **+154**; einzeln: Erfolgsaufschlag a +85, Sonderwertung +70, Bremsfaktor b +42, Untergrenze +18. | `test_all_rules_pay_out_154_points_in_a_closed_field_and_a_pays_most` |
| Woher die Punkte kommen | Im geschlossenen Feld zahlen a rund 64 und die Sonderwertung rund 68 Punkte je Spieler aus; die Sonderwertung löst 54-mal je Lauf aus. | `test_the_points_of_the_rules_are_the_posts_of_the_balance` |
| Lernende Jugend (350 Punkte unter ihrer Spitze, 12 % Aufholen je Saison) | ohne Gegenmittel **−130**, nur K aus Index und Alter −150, mit allen Regeln **+180**. | `test_learning_youth_deflates_without_remedies_and_inflates_with_all_rules` |
| Welche Regel trägt bei der Jugend? | Mit Erfolgsaufschlag a allein −3, mit Erst-DWZ-Korrektur r −43, mit Sonderwertung −140; alle Regeln außer a +49. | `test_the_success_surcharge_a_does_most_of_the_work_for_the_youth` |
| Bilanz ohne Gegenmittel | Zufälliger Abgang: Erst-DWZ der Neuen **+16**, Wertung der Gehenden +12 über der Stärke, Drift +25. Die Schwächsten gehen: **+88** und +57, Drift **+172** (Abgangsquote mal Differenz mal 40 Saisons ≈ 190). | `test_without_remedies_the_drift_is_entry_minus_exit` |
| Mit allen Regeln | Zufälliger Abgang **+303**, die Schwächsten gehen **+301**, das Feld lernt +163; Erst-DWZ der Neuen +247 bzw. +246 über ihrer Stärke. | `test_with_all_rules_both_scenarios_end_near_plus_300` |
| Sonderwertung | Das Feld lernt: ohne Gegenmittel **−115**, mit allen Regeln **+163**; im geschlossenen Feld allein +70 gegen ±0 ohne sie. | `test_the_special_rating_turns_the_sign_when_the_whole_field_learns` |
| Eintrittsregel (Kontrollversuch: ein Neuling gegen ein genau gewertetes Feld, 9 Partien) | Liegt der Neuling 100 / 300 / 500 Punkte unter dem Feld, überschätzt ihn die Leistung Ri um **−2 / +18 / +116** Punkte, wenn nur gewertet wird, wer mindestens 0,5 Punkte holt (Anteil der Gewerteten 0,99 / 0,86 / 0,50); ohne die Hürde liegt der Fehler bei 300 Punkten bei −27, mit dem Aufschlag r = 0,08 bei +64. | `test_the_threshold_of_half_a_point_selects_the_lucky_newcomers` |
| Anhebung (Beginn Saison 21) | Steigung vor ihr 1,02, **unmittelbar danach 0,67** (Jugend 0,60), am Ende der Saison 0,81, fünf Saisons später 1,01. | `test_the_uplift_compresses_the_scale_and_the_fixed_expectation_stretches_it_back` |
| Niveau nach der Anhebung | Sprung der Skalenlage bei Stärke 1500 um **+137**; am Ende bleiben +104 (zufälliger Abgang) bzw. +87 (Jugend) gegen den Lauf ohne Anhebung. | `test_the_uplift_jumps_the_level_by_about_137_and_two_thirds_to_a_half_remain` |
| Anker (Beitritt nach 2 / 30 Saisons, Wertung vier Saisons später) | Geschlossenes Feld **+82 → +171**, Jugend +115 → +204, das Feld lernt **+71 → +155**; Fehler jedes Mittels ±6 bis ±7. | `test_the_anchor_reads_the_scale_level_with_noise` |
| Teilnahme und Altersverteilung | Jugend 180 im Grundfall, 153 mit Aktivität und Mannschaftsturnier, 277 mit Altersverteilung, 250 mit beidem; Schwächste gehen 301, 274, 301, 268: Größen ändern sich, das Vorzeichen nie. | `test_participation_and_ages_shift_the_size_but_not_the_sign` |
| Was keinen Effekt hat | Die gedruckte Tabelle P(D) und das numpy-Rundungsverfahren ändern den Drift nicht messbar (geschlossen 154,1 und 153,9 gegen 153,9; Schwächste gehen 300,8 und 301,2 gegen 301,2). | `test_table_and_rounding_have_no_measurable_effect` |
| Beispielturnier der WO | Erst-DWZ der Neuen J bis N 1476 / 1180 / 2328 / 1436 / 1494 (WO: 1476 / 1179 / 2329 / 1436 / 1494); DWZ-Änderungen A bis H 25 / −19 / −22 / 23 / −49 / −8 / 90 / −7 (WO: … / −8). | `test_beispielturnier_erst_dwz_der_zusatzspieler_und_gesamtiteration`, `test_beispielturnier_dwz_aenderungen_der_etablierten` |

Die Experimente 8 und 9 der App (Teilnahme, Alter, Tabelle, Rundung) sind Zeilen derselben Tabelle; die Buchhaltung eines gezeigten Laufs steht im Wasserfall der App.

## Ehrliche Grenzen

- **Keine Aussage über die reale DWZ.** Alle Zahlen sind Simulation; Anteil, Abstand und Tempo der Lernenden sind frei gewählt, die reale Jugendbewegung ist eine andere. Dass die Gegenmittel **überschießen**, ist
  eine Aussage über die Mechanik in diesem Modell, nicht über die DWZ oder die Wirkung der Anhebung.
- **Das Spielmodell ist die Wertung:** Die Partien folgen der Normalverteilung der DWZ; echte Partien weichen davon ab. Remis haben einen festen Anteil (30 % bei Gleichstand).
- **Nicht modelliert:** Zweikämpfe (nur mit beidseitiger DWZ), Hybridturniere, FIDE-Elo oder fremd-nationale Wertung als erste DWZ (Index 6), Vereinslose („textuelle Teilnehmer“), die Vorhaltefrist von 5 Jahren,
  Neuberechnungen bei nachgereichten Turnieren und Mannschaftskämpfe als Brettpaarungen (hier ein Turnier zu 7 Runden); keine Zeitkontrollen.
- **Index und Rundung:** Der Index zählt die Auswertungen (WO 2.2.4); das Rundungsverfahren von Rn nennt die WO nicht, die Demo rundet kaufmännisch (ohne messbaren Effekt, Experiment 9).
- **Die Anhebung ist modelliert, nicht bewertet:** Das Modell zeigt Stauchung, Rückdehnung und den Niveausprung; ob 0,64 sachgerecht ist, kann es nicht sagen. In den Deflations-Szenarien ist die Steigung ohne
  Gegenmittel kleiner als 1, nicht größer; eine Stauchung begründen sie also nicht.
- **Der Anker unterliegt selbst den Regeln** (a, b, Sonderwertung) und rauscht um etwa ±40 Punkte je Lauf; Verschiebungen unter 20 Punkten sind nicht zu deuten.
- **Quelle:** Die Regeln folgen der Online-Fassung und dem PDF der Wertungsordnung (gelesen am 2026-11-02); die Übersichtstabelle Anhang 3.1.10 ist in der Text-Extraktion nicht stimmig und wird nicht verwendet;
  die Wahrscheinlichkeitstabelle Anhang 2.1 ist nur in Stichproben eingesehen. Begründung der Anhebung: nur die DSB-Meldung vom 28.07.2026 (FIDE-Anhebung 2024 und „seit Jahren beobachtete Deflation im unteren
  DWZ-Bereich“).
- **Modellgröße:** Feld von 20 bis 100, 60 Läufe je Zelle, ein Generator; Unterschiede unter etwa 15 Punkten sind Rauschen.

## Verifikation

Die Suite hat 248 Tests (Laufzeit knapp 2 Minuten, seriell).

- **Orakel** (`tests/test_oracle_dwz.py`, 24 Tests): Gewinnerwartung gegen `math.erf`, K und Folge-DWZ gegen eine Regelschleife mit der Tabelle aus der WO, Turnierleistung gegen `scipy.optimize.brentq`, Gesamtiteration gegen
  `scipy.optimize.root` der gekoppelten Gleichungen, eine ganze Turnierauswertung (mit und ohne Sonderwertung) gegen eine Partie-für-Partie-Rechnung, die Erst-DWZ mit Restpartien und Verfall, die Sonderwertung gegen
  die Formel der erzeugten Punkte und die Bilanz der Wertungssumme (exakt) auf neun Modellvarianten.
- **Literatur** (`tests/test_literatur_dwz.py`, 10 Tests): das durchgerechnete Beispielturnier der WO (Anfangswerte Rp, Endwerte Ri und R1, Turnierleistungen, Erwartungen, K, a, b, DWZ-Änderungen), Scoresheets H und B,
  Werte der Wahrscheinlichkeitstabelle, Anhebung 1528 / 1784 / 2040, Rechner-Beispiel. Quellen und Stand der Einsicht im Docstring.
- **Eigenschaften** (`tests/test_eigenschaften_dwz.py`, 14 Tests): Bilanzidentität auf Zufallsmodellen, Verschiebung, Spiegelung, Umnummerieren, Monotonie, K0 nicht steigend, Untergrenze und Anhebung, Erst-DWZ.
- **Einheiten:** Regeln (`tests/test_rules.py`), Feldmodell (`tests/test_field.py`), Auswertung und Verdikte (`tests/test_evaluation.py`), Presets (`tests/test_presets.py`), Generator (`tests/test_generator.py`).
- **Experimente:** `tests/test_experiments_data.py` (Datei gegen frischen Lauf je Konfiguration) und `tests/test_claims.py` (jede README-Zahl).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Reglerän­derung hebt die
  Markierung auf), ausblendbare Regler (KEPT-Muster), PDF-Export, Footer und Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `dwz_rules.py` | Regeln der WO als reine Funktionen: P(D), K, Folge-DWZ, Erst-DWZ, Gesamtiteration, Turnierleistung, Anhebung |
| `dwz_field.py` | Offenes Feld: Saisons, Turniere, Auswertung, Abgang und Zugang, Anker, Bilanz |
| `dwz_scenario.py` | Reglerwerte (`Settings`) und Modell |
| `dwz_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `dwz_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), KEPT-Regler, Zufalls-Seed-Button |
| `dwz_evaluation.py` | Kennzahlen, Bilanz, Analyse des Laufs, Katalog der Experimente, Verdikte |
| `dwz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `dwz_pdf_export.py` | PDF-Export (fpdf2) |
| `dwz_layout.py` | Fester Kopf (Stylesheet) |
| `turnier_generator.py` | Gemeinsamer Generator der Linie (Version 1, wortgleich): SplitMix64-Strom |
| `data/experiments.json` | Vorberechnete Experimente (54 Konfigurationen, je 60 Läufe) |
| `tools/build_experiments.py` | Baut `data/experiments.json` neu |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Literatur, Eigenschaften, Einheiten, Experimente, Claims, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Die frühere Formel `800/(E + n)`** (Fassung vor 2026): Die Quellenlage besteht aus Sekundärquellen, und die Wertungsordnung ab dem 8. Juni 2026 hat sie ersetzt.
- **Eine Aussage über die reale DWZ-Verteilung** oder über die Wirkung der Anhebung: Dafür fehlen Daten; die Demo zeigt Mechanik.
- **Vergleich mit anderen Verfahren über die Zeit** (Glicko, TrueSkill, WHR): gehört zum Stück „Welches Verfahren, gemessen woran?“ der Linie.
- **Ein Regler für den Remisanteil, die Streuung der Stärken oder die Normalverteilung:** feste Annahmen, damit die Experimente vergleichbar bleiben.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Experimentdaten neu bauen (nach jeder Änderung an Regeln oder Feldmodell; sonst schlägt `tests/test_experiments_data.py` fehl):

```bash
python tools/build_experiments.py 8
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
