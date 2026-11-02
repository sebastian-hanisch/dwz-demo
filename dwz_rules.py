"""Regeln der DSB-Wertungsordnung (gültig ab 8. Juni 2026) als reine Funktionen auf Feldern (numpy), ohne Zufall und ohne Zustand.

Fundstellen (Artikel der Wertungsordnung, `WO`): Gewinnerwartung 3.3 und Anhang 2.1; Entwicklungsfaktor K 3.5.1 und Anhang 1.3; Folge-DWZ 3.5.2; Erst-DWZ 3.4 und
Anhang 1.1 bis 1.2.2; Turnierleistung 3.6; Anhebung Anhang 4. Alle Wertungen sind DWZ-Punkte."""

import math

import numpy as np
from scipy.special import ndtr, ndtri

SIGMA = 200.0 * math.sqrt(2.0)       # WO 3.3.1: Standardabweichung der Differenz D, rund 282,8
FLOOR = 1100.0                       # WO 3.5.2 und 3.4.4.2: DWZ-Untergrenze und kleinste Erst-DWZ
K_MAX = 80.0                         # WO Anhang 1.2.1: Höchstwert des K-Faktors (Kmax)
A_DIVISOR = 30.0                     # t: Jugend-Aufschlag a = (2000 − R0)/t
A_ADULT = 4.0                        # a1 = a2: Erfolgsaufschlag Junioren und Erwachsene ab 21 Jahren unter 1600
B_SHIFT = 800.0                      # u: Bremsfaktor b = (R0 + u)/(1600 + u)
LIMIT = 1600.0                       # Wertungsgrenze von Erfolgsaufschlag und Bremsfaktor
YOUTH_AGE, JUNIOR_AGE = 20, 25       # Jugendliche bis 20 Jahre, Jugendliche/Junioren bis 25 Jahre (Gruppe von K0)
FIRST_R = 0.08                       # r: Erst-DWZ-Korrektur (Anhang 1.2.2)
MIN_GAMES, MIN_POINTS = 5, 0.5       # Zusatzstufen (WO 3.4.1): mindestens 5 Partien gegen Auswahl I und 0,5 Punkte
SPECIAL_MARGIN = 300.0               # Sonderwertung (WO 3.6.3)
PERF_MIN_GAMES = 5                   # Turnierleistung ab 5 wertbaren Partien (WO 3.6.1)
REST_YEARS = 2.0                     # Restpartien verfallen nach 2 Jahren (WO 4.1.1)

# Grundwert K0 nach Index 1 bis 10 und > 10 (WO Anhang 1.3); Zeilen: bis 25 Jahre unter DWZ 2200 | ab 26 Jahren unter DWZ 2000 | darüber
K0_TABLE = np.array([[60, 60, 48, 46, 44, 42, 40, 38, 36, 34, 32],
                     [60, 60, 44, 42, 40, 38, 36, 34, 32, 30, 28],
                     [60, 60, 41, 39, 37, 35, 33, 31, 29, 27, 25]], dtype=float)


def win_prob(d, table: bool = False):
    """Gewinnerwartung P(D) = Φ(D/σ) (WO 3.3.1). `table`: wie die gedruckte Tabelle, D ganzzahlig und p auf drei Stellen (Anhang 2.1)."""
    d = np.asarray(d, dtype=float)
    if table:
        return np.round(ndtr(np.round(d) / SIGMA), 3)
    return ndtr(d / SIGMA)


def rating_difference(p):
    """Umkehrung D(P) (Anhang 2.1): σ·Φ⁻¹(p), für p unter 0,005 und über 0,995 abgeschnitten (rund ±729)."""
    return SIGMA * ndtri(np.clip(p, 0.005, 0.995))


def k_parts(r0, age, idx, win):
    """Grundwert K0, Erfolgsaufschlag a und Bremsfaktor b (WO Anhang 1.3), je Spieler.
    `win` ist wahr, wenn W ≥ We: nur dann gibt es a; b nur bei W < We. Index ≥ 1 (Spieler mit Wertung)."""
    r0, age, idx, win = (np.asarray(x) for x in (r0, age, idx, win))
    young = age <= JUNIOR_AGE
    row = np.where(young, np.where(r0 < 2200, 0, 2), np.where(r0 < 2000, 1, 2))
    k0 = K0_TABLE[row, np.minimum(idx, 11) - 1]
    a = np.where(win & (age <= YOUTH_AGE) & (r0 < 2000), (2000.0 - r0) / A_DIVISOR, np.where(win & (age > YOUTH_AGE) & (r0 < LIMIT), A_ADULT, 0.0))
    b = np.where((~win) & (r0 < LIMIT), (r0 + B_SHIFT) / (LIMIT + B_SHIFT), 1.0)
    return k0, a, b


def k_factor(r0, age, idx, win, use_a: bool = True, use_b: bool = True, fixed=None):
    """Entwicklungsfaktor K = K0·b + a, auf eine Dezimale gerundet und auf Kmax begrenzt (WO Anhang 1.3). Ergebnis (K, K0, a, b); `fixed`: fester K ohne a und b."""
    r0 = np.asarray(r0, dtype=float)
    if fixed is not None:
        n = r0.shape
        return np.full(n, float(fixed)), np.full(n, float(fixed)), np.zeros(n), np.ones(n)
    k0, a, b = k_parts(r0, age, idx, win)
    a = a if use_a else np.zeros_like(a)
    b = b if use_b else np.ones_like(b)
    return np.minimum(np.floor((k0 * b + a) * 10.0 + 0.5) / 10.0, K_MAX), k0, a, b


def round_rating(x, mode: str = "up"):
    """Ganzzahlige Rundung der Folge-DWZ: kaufmännisch („up“) oder numpy-Standard („even“); die WO nennt kein Verfahren (3.5.2)."""
    x = np.asarray(x, dtype=float)
    return np.floor(x + 0.5) if mode == "up" else np.rint(x)


def update(r0, k, w, we, floor=FLOOR, mode: str = "up"):
    """Folge-DWZ Rn = R0 + K·(W − We), ganzzahlig, nicht unter der Untergrenze (WO 3.5.2). Ergebnis (Rn, ungerundet, gerundet vor der Untergrenze)."""
    raw = np.asarray(r0, dtype=float) + np.asarray(k, dtype=float) * (np.asarray(w, dtype=float) - np.asarray(we, dtype=float))
    rounded = round_rating(raw, mode)
    final = np.maximum(rounded, floor) if floor is not None else rounded
    return final, raw, rounded


def first_rating_correction(ri, r: float = FIRST_R):
    """Erst-DWZ R1 aus dem Iterationswert Ri (Anhang 1.2.2 und 3.4.4.2): ab 2000 unverändert, von 1100 bis 2000 `Ri + r(2000 − Ri)`, darunter
    `1100 + 9·r·Ri/11`; ein negatives Ri zählt als 0, die kleinste Erst-DWZ ist 1100."""
    ri = np.maximum(np.asarray(ri, dtype=float), 0.0)
    return np.where(ri >= 2000, ri, np.where(ri >= 1100, ri + r * (2000.0 - ri), 1100.0 + 9.0 * r * ri / 11.0))


def performance(owner, opp, score, n_players, full):
    """Turnierleistung je Spieler: die DWZ x mit We(x) = W über die Zeilen (owner, opp, score) (WO 3.6.1, Anhang 1.1.4), mit fiktivem Remis gegen den Gegnerschnitt
    bei `full` (100 %, WO 3.6.2). Negative Werte zählen als 0. Bisektion; Spieler ohne Zeilen bekommen einen bedeutungslosen Wert."""
    owner = np.asarray(owner, dtype=int)
    opp = np.asarray(opp, dtype=float)
    score = np.asarray(score, dtype=float)
    ng = np.bincount(owner, minlength=n_players)
    w = np.bincount(owner, score, minlength=n_players)
    mean_opp = np.bincount(owner, opp, minlength=n_players) / np.maximum(ng, 1)
    target = np.where(full, w + 0.5, w)
    lo, hi = np.full(n_players, -2000.0), np.full(n_players, 6000.0)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        we = np.bincount(owner, ndtr((mid[owner] - opp) / SIGMA), minlength=n_players) + np.where(full, ndtr((mid - mean_opp) / SIGMA), 0.0)
        up = we < target
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return np.maximum(0.5 * (lo + hi), 0.0)


def initial_values(owner, opp_idx, fixed, score, member, stage1, n_players):
    """Anfangswerte Rp = Rc + D(p) der Gesamtiteration (WO Anhang 1.1.1): Die 1. Zusatzstufe zählt nur Partien gegen Etablierte und Restpartien (feste Gegner), die 2. Zusatzstufe
    darüber hinaus die Partien gegen die 1. Zusatzstufe mit deren Rp; Partien innerhalb der 2. Stufe zählen nicht. Rückgabe: Rp je Platz (0 für Nichtmitglieder)."""
    owner = np.asarray(owner, dtype=int)
    opp_idx = np.asarray(opp_idx, dtype=int)
    fixed = np.asarray(fixed, dtype=float)
    score = np.asarray(score, dtype=float)
    is_var = opp_idx >= 0
    safe = np.where(is_var, opp_idx, 0)
    stage1 = np.asarray(stage1, dtype=bool)
    rows1 = ~is_var
    n1 = np.bincount(owner[rows1], minlength=n_players).astype(float)
    rc1 = np.bincount(owner[rows1], fixed[rows1], minlength=n_players) / np.maximum(n1, 1.0)
    rp1 = rc1 + rating_difference(np.bincount(owner[rows1], score[rows1], minlength=n_players) / np.maximum(n1, 1.0))
    rp = np.where(stage1, rp1, 0.0)
    rows2 = ~is_var | stage1[safe]
    opp_r = np.where(is_var, rp[safe], fixed)
    n2 = np.bincount(owner[rows2], minlength=n_players).astype(float)
    rc2 = np.bincount(owner[rows2], opp_r[rows2], minlength=n_players) / np.maximum(n2, 1.0)
    rp2 = rc2 + rating_difference(np.bincount(owner[rows2], score[rows2], minlength=n_players) / np.maximum(n2, 1.0))
    return np.where(np.asarray(member, dtype=bool), np.where(stage1, rp1, rp2), 0.0)


def total_iteration(owner, opp_idx, fixed, score, member, n_players, stage1=None, tol: float = 0.1, max_iter: int = 400):
    """Gesamtiteration der Zusatzspieler (WO Anhang 1.1): Zeilen (owner, opp_idx, fixed, score); `opp_idx` ≥ 0 verweist auf einen Mitspieler der Iteration (seine laufende
    Schätzung zählt als Gegner-DWZ), −1 auf eine feste Gegner-DWZ `fixed` (Etablierte, Restpartien). Anfangswert Rp (`initial_values`); je Schritt D' = D(p') mit
    p' = 0,5 + (W − We)/n, bei 100 % mit fiktivem Remis gegen den Gegnerschnitt Rc aus den Anfangswerten (WO 3.4.1.3). Abbruch bei max |D'| < tol. Rückgabe (Ri, letzter Schritt).
    `stage1`: Mitglieder der 1. Zusatzstufe (ohne Angabe gelten alle Mitglieder als 1. Stufe)."""
    owner = np.asarray(owner, dtype=int)
    opp_idx = np.asarray(opp_idx, dtype=int)
    fixed = np.asarray(fixed, dtype=float)
    score = np.asarray(score, dtype=float)
    member = np.asarray(member, dtype=bool)
    stage1 = member if stage1 is None else np.asarray(stage1, dtype=bool)
    is_var = opp_idx >= 0
    safe = np.where(is_var, opp_idx, 0)
    n = np.bincount(owner, minlength=n_players).astype(float)
    w = np.bincount(owner, score, minlength=n_players)
    x = initial_values(owner, opp_idx, fixed, score, member, stage1, n_players)
    rc = np.bincount(owner, np.where(is_var, x[safe], fixed), minlength=n_players) / np.maximum(n, 1.0)       # Gegnerschnitt mit den Rp der Mitspieler
    full = member & (w >= n - 1e-9)
    n_eff, w_eff = n + full, w + 0.5 * full
    step = np.zeros(n_players)
    for _ in range(max_iter):
        opp = np.where(is_var, x[safe], fixed)
        we = np.bincount(owner, ndtr((x[owner] - opp) / SIGMA), minlength=n_players) + np.where(full, ndtr((x - rc) / SIGMA), 0.0)
        step = np.where(member, rating_difference(0.5 + (w_eff - we) / np.maximum(n_eff, 1.0)), 0.0)
        x = x + step
        if np.abs(step).max() < tol:
            break
    return x, float(np.abs(step).max())


def uplift(r):
    """Einmalige Anhebung der DWZ 2026 (WO Anhang 4): +400 bis 1000, `0,64·R + 760` bis 2110, darüber unverändert; ganzzahlig."""
    r = np.asarray(r, dtype=float)
    return np.floor(np.where(r <= 1000, r + 400.0, np.where(r <= 2110, 0.64 * r + 760.0, r)) + 0.5)
