"""Literatur-Referenzwerte für die DWZ (ORACLE-PLAYBOOK §5.1). Nur Werte, die in einer Quelle eingesehen wurden (Stand der Einsicht: 2026-11-02).

Quelle der meisten Werte: Deutscher Schachbund, *Wertungsordnung*, gültig ab 8. Juni 2026, PDF `WO_2026.pdf` (37 Seiten; von schachbund.de/wertungsordnung.html geladen und mit
pypdf als Text gelesen). Dort steht in Anhang 3.1 (Seiten 25 bis 35) ein vollständig durchgerechnetes Beispielturnier mit 16 Spielern A bis R („Auswertung eines Beispielturniers“,
Stellgrößen auf den Standardwerten). Genutzt werden:

1. Anhang 3.1.2 bis 3.1.4 (Seiten 26 bis 29): Anfangswerte Rp der Zusatzspieler J, K, L, M, N: 1544 / 1255 / 2554 / 1417 / 1646; Endwerte der Gesamtiteration Ri nach 20 Schritten:
   1430 / 1108 / 2329 / 1387 / 1450; korrigiert zur Erst-DWZ R1: 1476 / 1179 / 2329 / 1436 / 1494 (Tabelle „Zusammenfassung der Gesamtiteration“, Seite 29).
2. Anhang 3.1.6 und 3.1.7 (Seiten 30 und 31): Turnierleistungen Rh der Etablierten A bis H: 2331 / 1857 / – / 2007 / 1580 / 1574 / 1961 / 1227; Spieler G (DWZ 1640, Rh 1961) löst die Sonderwertung
   aus (Rh > R0 + 300); C hat weniger als 5 Partien, keine Rh. A mit 7 aus 7: fiktives Remis gegen den Gegnerschnitt 1809 (Seite 30).
3. Anhang 3.1.8 und 3.1.9 (Seiten 31 und 32): Punkterwartungen We der Etablierten A bis H: 5,990 / 3,088 / 1,873 / 3,818 / 4,160 / 3,241 / 1,630 / 1,308; Erfolgsaufschlag a (nur D: 1,667), Bremsfaktor b
   (nur H: 0,875), Grundwert K0: 25 / 32 / 25 / 32 / 42 / 34 / 48 / 28, Entwicklungsfaktor K: 25,0 / 32,0 / 25,0 / 33,7 / 42,0 / 34,0 / 48,0 / 24,5.
4. Anhang 3.1.10 (Seite 33): DWZ-Änderungen ΔR (ganzzahlig): A +25, B −19, C −22, D +23, E −49, F −8, G +90, H −8. Die Spalte Rn derselben Tabelle ist in der Text-Extraktion nicht stimmig
   (B: 2100 − 19 ergibt 2081, die Tabelle zeigt 2076) und wird nicht verwendet.
5. Anhang 3.1.12 und Seite 35: Scoresheet Spieler H (DWZ 1300, Index 77, We 1,308, K 24,5, neue DWZ 1292) und Beispiel Spieler B (Junior, K 32,0, W 2,5, We 3,088, ΔR −18,816, neue DWZ 2081).
6. Anhang 4 (Seite 36) und die DSB-Meldung vom 28.07.2026 (schachbund.de/news/die-neue-dwz-abfrage-ist-online-und-die-dwz-ist-angehoben.html): Anhebung 1200 → 1528, 1600 → 1784,
   2000 → 2040, ab 2110 keine Änderung.
7. Seite 8 (WO 3.3): P(−100) = 1 − 0,638 = 0,362 (Beispiel unter der Wahrscheinlichkeitstabelle).
8. Drittanbieter (kein DSB-Text, nur Gegenprobe): Rechner auf deutschland-rechner.de (abgerufen 2026-11-02): Erwachsener DWZ 1650, Index 13, W − We = +0,619: K = 28,0, Rn = 1667.

Die WO rechnet die Iteration mit ganzzahligen Differenzen D; die Demo rechnet stetig und bricht bei einem Schritt unter 0,1 ab. Die Tests lassen deshalb bei Ri und Rh wenige Punkte Abstand zu.

Nicht eingesehen / offen: die Wahrscheinlichkeitstabelle Anhang 2.1 als Ganzes (nur die oben genannten Werte); das vollständig durchgerechnete Beispiel der Wertungskommission („kann als Datei angefordert werden“);
das Rundungsverfahren von Rn (die WO nennt keines, 3.5.2)."""

import numpy as np
import pytest

import dwz_rules as W
from conftest import make_state, run_tournament, small

NAMES = "ABCDEFGHJKLMN"
IDX = {c: i for i, c in enumerate(NAMES)}
# Etablierte A bis H: DWZ, Index, Alter (WO Anhang 3.1.9); E führt eine FIDE-Zahl als DWZ (Index 6)
RATING = [2220, 2100, 2000, 1950, 1800, 1700, 1640, 1300]
INDEX = [31, 55, 25, 15, 6, 8, 3, 77]
AGE = [40, 22, 55, 16, 20, 43, 24, 66]
# Partien des Beispielturniers (Spieler, Gegner, Punkte des ersten), aus Anhang 3.1.1 bis 3.1.8 gelesen; Partien von P, Q und R gegen die Auswahl I werden nur Restpartien und fehlen hier
GAMES = [("A", "B", 1), ("A", "C", 1), ("A", "D", 1), ("A", "E", 1), ("A", "F", 1), ("A", "G", 1), ("A", "J", 1),
         ("B", "E", .5), ("B", "G", 1), ("B", "K", 1), ("B", "L", 0),
         ("C", "D", 0), ("C", "L", 0), ("C", "N", 1),
         ("D", "E", 1), ("D", "G", .5), ("D", "H", 1), ("D", "J", 1), ("D", "L", 0),
         ("E", "F", 0), ("E", "G", 0), ("E", "H", 1), ("E", "J", .5), ("E", "L", 0), ("E", "N", 1),
         ("F", "G", 0), ("F", "H", 1), ("F", "K", .5), ("F", "L", 0), ("F", "N", .5),
         ("G", "J", 1),
         ("H", "J", 0), ("H", "K", .5), ("H", "N", .5),
         ("J", "M", .5), ("K", "M", 0), ("K", "N", 0), ("L", "N", 1), ("M", "N", .5)]
# Restpartien der Zusatzspieler (Gegner-DWZ, Punkte), Anhang 3.1.2 und 3.1.3
REST = {"J": [(1500, 0)], "K": [(1300, .5), (1250, 0), (1400, 0), (1300, 0)], "L": [(1400, 1)], "M": [(1300, 1), (1500, 0), (1600, 0), (1300, .5)]}
PLAYED = {"A": 7, "B": 2.5, "C": 1, "D": 4.5, "E": 3, "F": 3, "G": 3.5, "H": 1}                    # erzielte Punkte W der Etablierten (Seite 26)


def _example(special=True):
    m = small(n=13, special=special)
    rating = np.array(RATING + [0.0] * 5, dtype=float)
    rated = np.array([True] * 8 + [False] * 5)
    st, dr = make_state(m, rating, np.full(13, 1500.0), age=np.array(AGE + [30] * 5), idx=np.array(INDEX + [0] * 5), rated=rated)
    for c, rows in REST.items():
        st.rest[IDX[c]] = [(float(r), float(s), 0.0) for r, s in rows]
    games = [(IDX[a], IDX[b], float(w)) for a, b, w in GAMES]
    acc = run_tournament(st, dr, m, games, part=np.r_[np.ones(13, bool), False], now=0.0)
    return st, acc


def test_beispielturnier_punkte_der_etablierten_stimmen_mit_der_tabelle():
    got = {c: 0.0 for c in "ABCDEFGH"}
    for a, b, w in GAMES:
        if a in got:
            got[a] += w
        if b in got:
            got[b] += 1 - w
    assert got == PLAYED


def test_beispielturnier_erst_dwz_der_zusatzspieler_und_gesamtiteration():
    st, _acc = _example()
    r1 = {c: float(st.R[IDX[c]]) for c in "JKLMN"}
    assert all(st.rated[IDX[c]] and st.idx[IDX[c]] == 1 for c in "JKLMN")
    expect = {"J": 1476, "K": 1179, "L": 2329, "M": 1436, "N": 1494}
    for c, v in expect.items():
        assert r1[c] == pytest.approx(v, abs=3), (c, r1[c], v)
    # Anhang 3.1.4: Rückrechnung von R1 auf den Endwert Ri (r = 0,08): Ri = (R1 − 0,08·2000)/0,92, bei R1 ≥ 2000 gleich R1
    ri_wo = {"J": 1430, "K": 1108, "L": 2329, "M": 1387, "N": 1450}
    for c, v in ri_wo.items():
        ri = r1[c] if r1[c] >= 2000 else (r1[c] - 160.0) / 0.92
        assert ri == pytest.approx(v, abs=3), (c, ri, v)


def test_anfangswerte_rp_der_gesamtiteration_nach_anhang_1_1_1():
    """Rp = Rc + D(p): J 1544, K 1255, L 2554 (100 %: D = 729), M 1417, N 1646 (Seite 26 und 27)."""
    # feste Gegner (Etablierte, Restpartien) und Mitspieler der Iteration; J, K, L, M, N = 0 bis 4
    rows = []
    for c, opp, sc in (("J", 1500, 0), ("J", 2220, 0), ("J", 1950, 0), ("J", 1800, .5), ("J", 1640, 0), ("J", 1300, 1),
                       ("K", 1300, .5), ("K", 1250, 0), ("K", 1400, 0), ("K", 1300, 0), ("K", 2100, 0), ("K", 1700, .5), ("K", 1300, .5),
                       ("L", 1400, 1), ("L", 2100, 1), ("L", 2000, 1), ("L", 1950, 1), ("L", 1800, 1), ("L", 1700, 1),
                       ("M", 1300, 1), ("M", 1500, 0), ("M", 1600, 0), ("M", 1300, .5),
                       ("N", 2000, 0), ("N", 1800, 0), ("N", 1700, .5), ("N", 1300, .5)):
        rows.append(("JKLMN".index(c), -1, float(opp), float(sc)))
    for c, o, sc in (("M", "J", .5), ("M", "K", 1), ("N", "K", 1), ("N", "L", 0)):
        rows.append(("JKLMN".index(c), "JKLMN".index(o), 0.0, float(sc)))
    owner, opp_idx, fixed, score = (np.array(col) for col in zip(*rows))
    member = np.ones(5, dtype=bool)
    stage1 = np.array([True, True, True, False, False])
    rp = W.initial_values(owner, opp_idx.astype(int), fixed, score, member, stage1, 5)
    for got, want in zip(rp, (1544, 1255, 2554, 1417, 1646)):
        assert got == pytest.approx(want, abs=1.5)


def test_turnierleistungen_und_sonderwertung_des_beispielturniers():
    """Rh von A (fiktives Remis bei 7 aus 7) und G (Ausnahmespieler); G löst die Sonderwertung als Einziger aus."""
    st, acc = _example()
    assert acc["special_cases"] == 1
    r1 = {c: float(st.R[IDX[c]]) for c in "JKLMN"}
    # A: Gegner B C D E F G J mit 2100 2000 1950 1800 1700 1640 und R1(J)
    opp_a = np.array([2100, 2000, 1950, 1800, 1700, 1640, r1["J"]], dtype=float)
    rh_a = float(W.performance(np.zeros(7, dtype=int), opp_a, np.ones(7), 1, np.array([True]))[0])
    assert rh_a == pytest.approx(2331, abs=3)
    opp_g = np.array([2220, 2100, 1950, 1800, 1700, r1["J"]], dtype=float)
    rh_g = float(W.performance(np.zeros(6, dtype=int), opp_g, np.array([0, 0, .5, 1, 1, 1], dtype=float), 1, np.array([False]))[0])
    assert rh_g == pytest.approx(1961, abs=3) and rh_g > 1640 + W.SPECIAL_MARGIN


def test_beispielturnier_dwz_aenderungen_der_etablierten():
    st, _acc = _example()
    delta = {c: float(st.R[IDX[c]]) - RATING[IDX[c]] for c in "ABCDEFGH"}
    for c, want in zip("ABCDEFGH", (25, -19, -22, 23, -49, -8, 90, -8)):
        assert delta[c] == pytest.approx(want, abs=1), (c, delta[c], want)
    assert (st.idx[:8] == np.array(INDEX) + 1).all()                                  # Anhang 3.1.10: der Index steigt um 1


def test_entwicklungsfaktoren_des_beispielturniers():
    won = np.array([True, False, False, True, False, False, True, False])          # W ≥ We aus Anhang 3.1.9: A, D, G
    k, k0, a, b = W.k_factor(np.array(RATING, dtype=float), np.array(AGE), np.array(INDEX), won)
    assert list(k0) == [25, 32, 25, 32, 42, 34, 48, 28]
    assert a[3] == pytest.approx(5 / 3, abs=1e-9) and a[[0, 1, 2, 4, 5, 6, 7]].sum() == 0
    assert b[7] == pytest.approx(0.875) and (b[:7] == 1).all()
    assert list(np.round(k, 1)) == [25.0, 32.0, 25.0, 33.7, 42.0, 34.0, 48.0, 24.5]


def test_scoresheet_spieler_h_und_beispiel_spieler_b():
    h_opp = np.array([1950, 1700, 1494, 1179, 1476, 1800], dtype=float)
    h_w = 1.0
    we = float(W.win_prob(1300 - h_opp, table=True).sum())
    assert we == pytest.approx(1.308, abs=0.0006)                                      # 0,011 + 0,079 + 0,246 + 0,666 + 0,267 + 0,039
    k = float(W.k_factor(np.array([1300.0]), np.array([66]), np.array([77]), np.array([h_w >= we]))[0][0])
    assert k == 24.5
    assert float(W.update(np.array([1300.0]), np.array([k]), np.array([h_w]), np.array([we]))[0][0]) == 1292
    b_opp = np.array([2220, 1800, 1961, 1179, 2329], dtype=float)
    b_we = float(W.win_prob(2100 - b_opp, table=True).sum())
    assert b_we == pytest.approx(3.088, abs=0.0006)                                    # 0,336 + 0,856 + 0,688 + 0,999 + 0,209
    kb = float(W.k_factor(np.array([2100.0]), np.array([23]), np.array([56]), np.array([2.5 >= b_we]))[0][0])
    assert kb == 32.0 and float(W.update(np.array([2100.0]), np.array([kb]), np.array([2.5]), np.array([b_we]))[0][0]) == 2081


def test_wahrscheinlichkeitswerte_aus_der_wo():
    assert float(W.win_prob(-100.0, table=True)) == pytest.approx(1 - 0.638, abs=1e-9)
    for d, p in ((120, .664), (220, .782), (270, .830), (420, .931), (520, .967), (259, .820), (744, .996), (-120, .336), (300, .856), (139, .688), (921, .999), (-229, .209)):
        assert float(W.win_prob(d, table=True)) == pytest.approx(p, abs=1e-9), d


def test_anhebung_nach_anhang_4_und_der_dsb_meldung():
    got = W.uplift(np.array([1200.0, 1600.0, 2000.0, 2110.0, 2500.0]))
    assert list(got[:3]) == [1528, 1784, 2040] and got[4] == 2500


def test_rechner_beispiel_erwachsener_1650():
    k = float(W.k_factor(np.array([1650.0]), np.array([40]), np.array([13]), np.array([True]))[0][0])
    assert k == 28.0
    assert float(W.update(np.array([1650.0]), np.array([k]), np.array([1.0]), np.array([1.0 - 0.619]))[0][0]) == 1667
