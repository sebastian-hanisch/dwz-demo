"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

BASE = 1500.0                               # Mitte der Skala: Startwertung der Gründer, Stärke des Ankers (Konvention)
POP_SD = 200.0                              # Streuung der wahren Stärken im Feld (Punkte): Gründer und Zugänge
BURN_IN = 8                                 # Saisons ohne Zu- und Abgang vor der Messung (die Wertungen lernen die Stärken)
DRAW_RATE = 0.3                             # Remisanteil bei Gleichstand: d = 0,3·4p(1−p); die Punkterwartung bleibt p

N_MIN, N_MAX, N_STEP, DEFAULT_N = 20, 100, 2, 60
SEASONS_MIN, SEASONS_MAX, DEFAULT_SEASONS = 10, 50, 40                        # gemessene Saisons nach der Einlaufzeit
CHURN_MIN, CHURN_MAX, CHURN_STEP, DEFAULT_CHURN = 0.0, 0.3, 0.05, 0.15        # Anteil, der je Saison geht (und ebenso viele kommen)
LEAVE = ("zufällig", "die Schwächsten")
DEFAULT_LEAVE = LEAVE[0]
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = 0, 400, 50, 350                     # Abstand der Neuen unter ihrer Spitzenstärke (Punkte); 0 = Neue kommen mit ihrer Stärke
RATE_MIN, RATE_MAX, RATE_STEP, DEFAULT_RATE = 0.05, 0.4, 0.01, 0.12           # Anteil des Abstands, den die Lernenden je Saison aufholen
TREND_MIN, TREND_MAX, TREND_STEP, DEFAULT_TREND = -5.0, 5.0, 0.5, 0.0         # Änderung der Stärke aller je Saison (Punkte)
KMODE = ("DWZ: Index, Alter, Wertung", "fester K")
KFIX_MIN, KFIX_MAX, DEFAULT_KFIX = 10, 60, 28
SCHED = ("zwei Turniere, alle spielen", "Aktivität und Mannschaftsturnier")
UPLIFT_MIN, DEFAULT_UPLIFT = 0, 20                                           # Saison (nach der Einlaufzeit), zu deren Beginn die Anhebung 2026 gilt
JOIN_MIN, DEFAULT_JOIN = 0, 2                                                # Saison (nach der Einlaufzeit), zu deren Beginn der Anker beitritt
SEED_MAX, DEFAULT_SEED = 999999, 23

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_SEASONS, EXP_CHURN = 60, 40, 0.15
EXP_MARKS = (10, 20, 40)                                                     # Saisons, nach denen die Drift berichtet wird
EXP_UPLIFT = 20                                                              # Anhebung im Experiment: zu Beginn von Saison 20 nach der Einlaufzeit
ANCHOR_ERAS = (2, 15, 30)                                                    # Saisons, die bis zum Beitritt des Ankers gespielt sind
ANCHOR_STAY = 4                                                              # Saisons, nach denen die Wertung des Ankers gelesen wird (mit der des Beitrittssaison 4)
EXPERIMENT_FILE = "data/experiments.json"
NOISE = 20.0                                                                 # Streuung (Punkte) der Drift eines einzelnen Laufs (Größenordnung, Experiment 2)

COLOR_RATING = "#2F6B65"
COLOR_TRUTH = "#e67e22"
COLOR_ANCHOR = "#8e6bb0"
COLOR_CREATED = "#c0392b"
PALETTE = ("#2F6B65", "#8e6bb0", "#d98c1f", "#c0392b", "#3d7ab5", "#7a8f3c", "#b0577e", "#5a5a5a")

_BASE = {"n": DEFAULT_N, "seasons": DEFAULT_SEASONS, "churn": DEFAULT_CHURN, "leave": DEFAULT_LEAVE, "gap": DEFAULT_GAP, "rate": DEFAULT_RATE, "trend": DEFAULT_TREND,
         "kmode": KMODE[0], "kfix": DEFAULT_KFIX, "use_a": True, "use_b": True, "use_r": True, "use_floor": True, "special": True, "sched": SCHED[0], "ages": False,
         "use_uplift": False, "uplift": DEFAULT_UPLIFT, "use_anchor": True, "join": DEFAULT_JOIN, "seed": DEFAULT_SEED}
_NO_REMEDY = {"kmode": KMODE[1], "use_a": False, "use_b": False, "use_r": False, "use_floor": False, "special": False}
PRESETS = {
    "Lernende Jugend, DWZ-Regeln": dict(_BASE),
    "Lernende Jugend, ohne Gegenmittel": {**_BASE, **_NO_REMEDY},
    "Geschlossenes Feld": {**_BASE, "churn": 0.0, "gap": 0},
    "Die Schwächsten gehen": {**_BASE, "leave": LEAVE[1], "gap": 0},
    "Das Feld lernt": {**_BASE, "gap": 0, "trend": 3.5},
    "Anhebung 2026": {**_BASE, "gap": 0, "use_uplift": True},
}
PRESET_HELP = {
    "Lernende Jugend, DWZ-Regeln": "Jede Saison gehen 15 % zufällig, Neue kommen 350 Punkte unter ihrer späteren Stärke und holen 12 % des Abstands je Saison auf; es gelten alle Regeln der Wertungsordnung. "
                                   "Die Gegenmittel überschießen: Die Skala steigt über die Fähigkeit hinaus.",
    "Lernende Jugend, ohne Gegenmittel": "Dasselbe Feld, aber fester K, keine Zuschläge, keine Erst-DWZ-Korrektur, keine Untergrenze, keine Sonderwertung: Die Lernenden sind unterbewertet und ziehen "
                                         "den Älteren Punkte ab, die Skala fällt (Deflation).",
    "Geschlossenes Feld": "Niemand geht, niemand kommt, alle Stärken sind fest. Trotzdem steigen die Wertungen: Erfolgsaufschlag, Bremsfaktor und Sonderwertung zahlen Punkte aus, die nirgends abgezogen werden.",
    "Die Schwächsten gehen": "Jede Saison gehen 15 % mit der niedrigsten Wertung; Neue kommen auf dem Niveau des Feldes. Die Erst-DWZ der Neuen liegt zu hoch (nur wer mindestens 0,5 Punkte holt, wird gewertet), die Skala steigt.",
    "Das Feld lernt": "Alle werden je Saison um 3,5 Punkte besser, Neue kommen auf dem Niveau des Feldes. Ein Wertungssystem ohne Anker sieht das nicht; hier gleicht die Sonderwertung es mehr als aus.",
    "Anhebung 2026": "Zu Beginn von Saison 20 gilt die einmalige Anhebung nach Anhang 4: +400 bis 1000, danach 0,64·R + 760 bis 2110. Die Steigung der Wertung auf die Stärke fällt auf rund 0,65 und kehrt in einigen Saisons zurück.",
}
