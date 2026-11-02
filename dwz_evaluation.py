"""Kennzahlen, Analyse des gezeigten Laufs, Experimente über die festen Läufe (SWEEP_SEEDS; vorberechnet in data/experiments.json) und die Verdikte.

Drift (Punkte) = Änderung des Mittels von (Wertung − wahrer Stärke) über die Gewerteten des Feldes, gegenüber dem Ende der Einlaufzeit. Positiv heißt Inflation (die Wertungen
liegen über der Fähigkeit), negativ Deflation. Die absolute Fähigkeit gibt es im Modell nur per Konvention (Stärke 1500 = Mittel der Gründer); nachprüfbar ist sie über den Anker,
einen Spieler konstanter Stärke. Die Bilanz zerlegt die Änderung der Wertungssumme in Eintritt, Abgang und die Punkte der einzelnen Regeln (exakt, siehe `balance`)."""

import functools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import dwz_constants as C
import dwz_field as F
from dwz_scenario import Settings, make_model

NAN = float("nan")
DATA_FILE = Path(__file__).resolve().parent / C.EXPERIMENT_FILE


def de(x: float, digits: int = 0) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def thou(x: float) -> str:
    """Ganzzahl mit Punkt als Tausendertrenner."""
    return f"{round(x):,}".replace(",", ".")


def signed(x: float, digits: int = 0) -> str:
    """Mit Vorzeichen und echtem Minuszeichen, damit „+46“ und „−147“ gleich breit lesbar sind."""
    if round(x, digits) == 0:
        return "±" + de(0.0, digits)
    s = f"{x:+.{digits}f}".replace(".", ",")
    return s.replace("-", "−")


# --- Der gezeigte Lauf ------------------------------------------------------------------------------------------------------------------------

BALANCE_KEYS = ("entry", "leave", "a", "b", "special", "rest", "rounding", "floor", "uplift", "anchor", "end")


def balance(run: F.Run) -> dict:
    """Änderung der Wertungssumme der Gewerteten vom Ende der Einlaufzeit bis zum Ende, in Punkten, zerlegt in Posten. Es gilt exakt
    end = entry − leave + a + b + special + rest + rounding + floor + uplift + anchor (Summe der Wertungen ändert sich nur durch diese Posten).
    entry: Erst-DWZ der neu Gewerteten, leave: Wertungen der gehenden Gewerteten, a/b: Punkte aus Erfolgsaufschlag und Bremsfaktor, special: aus der Sonderwertung,
    rest: aus ungleichem K0 und dem Rest, rounding/floor: aus Rundung und Untergrenze, uplift: Anhebung, anchor: Beitritt des Ankers."""
    b0 = run.model.burn
    after = slice(b0, None)
    s = run.series
    out = {"entry": float(s["entry"][after].sum()), "leave": float(s["leave"][b0 - 1:-1].sum()), "a": float(s["play_a"][after].sum()), "b": float(s["play_b"][after].sum()),
           "special": float(s["play_special"][after].sum()), "rest": float(s["play_rest"][after].sum()), "rounding": float(s["rounding"][after].sum()),
           "floor": float(s["floor_lift"][after].sum()), "uplift": float(s["uplift_gain"][after].sum()), "anchor": float(s["anchor_in"][after].sum())}
    out["end"] = float(s["total_end"][-1] - s["total_end"][b0 - 1])
    return out


def entry_leave_error(run: F.Run) -> tuple:
    """(Mittel von Erst-DWZ − wahrer Stärke der Neuen, Mittel von Wertung − Stärke der Gehenden) über die gemessenen Saisons; NaN ohne Neue bzw. Gehende."""
    b0 = run.model.burn
    s = run.series
    en = float(s["entry_n"][b0:].sum())
    ln = float(s["leave_n"][b0 - 1:-1].sum())
    return (float(s["entry_err"][b0:].sum()) / en if en else NAN, float(s["leave_err"][b0 - 1:-1].sum()) / ln if ln else NAN)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    run: F.Run
    drift: np.ndarray            # (Saisons,) Wertung − Fähigkeit gegenüber dem Ende der Einlaufzeit
    level: np.ndarray            # Änderung der mittleren wahren Stärke der Gewerteten
    rating: np.ndarray           # Änderung des Wertungsmittels
    slope: np.ndarray            # Steigung der Wertung auf die wahre Stärke (1 = Maßstab stimmt)
    fit: np.ndarray              # angepasste Wertung bei Stärke 1500 (Skalenlage)
    anchor: np.ndarray           # Wertung des Ankers, NaN außerhalb seiner Zeit
    direct: tuple                # (Drift, Steigung, fit) unmittelbar nach der Anhebung oder NaN
    bal: dict
    entry_err: float
    leave_err: float
    special_cases: float
    baseline: tuple              # (Steigung, fit1500) am Ende der Einlaufzeit


def analyse(s: Settings) -> Analysis:
    run = F.simulate(make_model(s), s.seed)
    b0 = run.model.burn
    sr = run.series
    base_bias, base_rating, base_true = sr["bias"][b0 - 1], sr["mean_rating"][b0 - 1], sr["mean_true"][b0 - 1]
    direct = run.direct
    direct = (direct[0] - base_bias, direct[1], direct[2]) if direct[0] == direct[0] else direct
    ent, lea = entry_leave_error(run)
    return Analysis(s, run, sr["bias"][b0:] - base_bias, sr["mean_true"][b0:] - base_true, sr["mean_rating"][b0:] - base_rating, sr["slope"][b0:], sr["fit"][b0:], sr["anchor"][b0:],
                    direct, balance(run), ent, lea, float(sr["special_cases"][b0:].sum()), (float(sr["slope"][b0 - 1]), float(sr["fit"][b0 - 1])))


def field_table(a: Analysis) -> list:
    """Spieler am Ende des Laufs: Platz, Alter, Index, Wertung, wahre Stärke und Abweichung (Punkte); Ungewertete ohne Wertung."""
    f = a.run.final
    rows = []
    for q in range(a.run.model.n + 1):
        if q == a.run.model.n and not f["active"][q]:
            continue
        rated = bool(f["rated"][q])
        rows.append({"Spieler": "Anker" if q == a.run.model.n else str(q + 1), "Alter": int(f["age"][q]), "Index": int(f["idx"][q]) if rated else 0,
                     "Wertung": round(float(f["rating"][q])) if rated else None, "wahre Stärke": round(float(f["true"][q])),
                     "Abweichung": round(float(f["rating"][q] - f["true"][q])) if rated else None})
    return rows


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

def _model(**kw) -> F.Model:
    return F.Model(**{"n": C.EXP_N, "seasons": C.EXP_SEASONS, "churn": C.EXP_CHURN, "gap": 0.0, **kw})


NONE = dict(kmode=C.KMODE[1], use_a=False, use_b=False, use_r=False, use_floor=False, special=False)        # ohne Gegenmittel: fester K 28, keine Zuschläge
BARE = dict(kmode=C.KMODE[0], use_a=False, use_b=False, use_r=False, use_floor=False, special=False)        # nur K aus Index, Alter und Wertung
YOUTH = dict(gap=350.0, rate=0.12)
CLOSED = dict(churn=0.0)
WEAK = dict(leave=C.LEAVE[1])


def catalog() -> list:
    """Alle Konfigurationen der Experimente: (Schlüssel, Gruppe, Beschriftung, Modell-Argumente)."""
    rows = []

    def add(key, group, label, **kw):
        rows.append((key, group, label, kw))

    # 1 Was die Gegenmittel erzeugen (geschlossenes Feld)
    add("c_none", "closed", "Ohne Gegenmittel (fester K)", **CLOSED, **NONE)
    add("c_bare", "closed", "Nur K aus Index, Alter, Wertung", **CLOSED, **BARE)
    add("c_a", "closed", "… plus Erfolgsaufschlag a", **CLOSED, **{**BARE, "use_a": True})
    add("c_b", "closed", "… plus Bremsfaktor b", **CLOSED, **{**BARE, "use_b": True})
    add("c_floor", "closed", "… plus Untergrenze 1100", **CLOSED, **{**BARE, "use_floor": True})
    add("c_special", "closed", "… plus Sonderwertung 300", **CLOSED, **{**BARE, "special": True})
    add("c_all", "closed", "Alle Regeln der WO", **CLOSED)
    # 2 Lernende Jugend
    add("y_none", "youth", "Ohne Gegenmittel (fester K)", **YOUTH, **NONE)
    add("y_bare", "youth", "Nur K aus Index, Alter, Wertung", **YOUTH, **BARE)
    add("y_a", "youth", "… plus Erfolgsaufschlag a", **YOUTH, **{**BARE, "use_a": True})
    add("y_r", "youth", "… plus Erst-DWZ-Korrektur r", **YOUTH, **{**BARE, "use_r": True})
    add("y_special", "youth", "… plus Sonderwertung 300", **YOUTH, **{**BARE, "special": True})
    add("y_noa", "youth", "Alle Regeln außer a", **YOUTH, use_a=False)
    add("y_all", "youth", "Alle Regeln der WO", **YOUTH)
    # 3 Bilanz und 4 Eintritt: Szenarien mit allen Regeln und ohne Gegenmittel
    for key, label, kw in (("closed", "Geschlossenes Feld", CLOSED), ("random", "Zufälliger Abgang", {}), ("weak", "Die Schwächsten gehen", WEAK), ("youth", "Lernende Jugend", YOUTH),
                           ("learns", "Das Feld lernt", dict(trend=3.5))):
        add(f"s_{key}_all", "scen", f"{label}, alle Regeln", **kw)
        add(f"s_{key}_none", "scen", f"{label}, ohne Gegenmittel", **kw, **NONE)
    # 5 Anhebung
    for key, label, kw in (("random", "Zufälliger Abgang", {}), ("youth", "Lernende Jugend", YOUTH)):
        add(f"u_{key}_off", "uplift", f"{label}, ohne Anhebung", **kw)
        add(f"u_{key}_on", "uplift", f"{label}, mit Anhebung", **kw, uplift_at=C.EXP_UPLIFT)
    # 6 Anker (Beitritt nach 2, 15, 30 Saisons)
    for key, label, kw in (("closed", "Geschlossenes Feld", CLOSED), ("weak", "Die Schwächsten gehen", WEAK), ("youth", "Lernende Jugend", YOUTH), ("learns", "Das Feld lernt", dict(trend=3.5))):
        for era in C.ANCHOR_ERAS:
            add(f"a_{key}_{era}", "anchor", f"{label}, Beitritt nach {era}", **kw, anchor_join=era)
    # 7 Was keinen Effekt hat
    for key, label, kw in (("closed", "Geschlossenes Feld", CLOSED), ("weak", "Die Schwächsten gehen", WEAK)):
        add(f"n_{key}_base", "noeffect", f"{label}, Normalverteilung, kaufmännisch", **kw)
        add(f"n_{key}_table", "noeffect", f"{label}, Tabelle der WO (D ganzzahlig, p auf 3 Stellen)", **kw, table=True)
        add(f"n_{key}_even", "noeffect", f"{label}, Rundung numpy-Standard", **kw, rounding="even")
    # 8 Teilnahme und Alter
    for key, label, kw in (("youth", "Lernende Jugend", YOUTH), ("weak", "Die Schwächsten gehen", WEAK)):
        add(f"v_{key}_base", "variants", f"{label}, Grundfall", **kw)
        add(f"v_{key}_sched", "variants", f"{label}, Aktivität und Mannschaftsturnier", **kw, sched=C.SCHED[1])
        add(f"v_{key}_ages", "variants", f"{label}, Altersverteilung", **kw, ages=True)
        add(f"v_{key}_both", "variants", f"{label}, beides", **kw, sched=C.SCHED[1], ages=True)
    return rows


def catalog_by_key() -> dict:
    return {key: (group, label, kw) for key, group, label, kw in catalog()}


RUN_FIELDS = ("d10", "d20", "d40", "level", "rating", "slope", "fit", "entry", "entry_n", "entry_err", "leave", "leave_n", "leave_err", "a", "b", "special", "rest", "rounding", "floor",
              "uplift", "anchor_in", "cases", "anchor4", "direct_slope", "direct_fit")


def summarize(run: F.Run) -> dict:
    """Kennzahlen eines Laufs für die Experimente (Punkte)."""
    m = run.model
    b0, s = m.burn, run.series
    bias0 = s["bias"][b0 - 1]
    last = m.seasons
    out = {f"d{mark}": float(s["bias"][b0 - 1 + min(mark, last)] - bias0) for mark in C.EXP_MARKS}
    out["level"] = float(s["mean_true"][-1] - s["mean_true"][b0 - 1])
    out["rating"] = float(s["mean_rating"][-1] - s["mean_rating"][b0 - 1])
    out["slope"] = float(s["slope"][-1])
    out["fit"] = float(s["fit"][-1] - s["fit"][b0 - 1])
    bal = balance(run)
    out.update({"entry": bal["entry"], "leave": bal["leave"], "a": bal["a"], "b": bal["b"], "special": bal["special"], "rest": bal["rest"], "rounding": bal["rounding"],
                "floor": bal["floor"], "uplift": bal["uplift"], "anchor_in": bal["anchor"], "entry_n": float(s["entry_n"][b0:].sum()), "entry_err": float(s["entry_err"][b0:].sum()),
                "leave_n": float(s["leave_n"][b0 - 1:-1].sum()), "leave_err": float(s["leave_err"][b0 - 1:-1].sum()), "cases": float(s["special_cases"][b0:].sum())})
    out["direct_slope"], out["direct_fit"] = run.direct[1], run.direct[2] - s["fit"][b0 - 1]
    out["anchor4"] = NAN
    if m.anchor_join is not None:
        out["anchor4"] = float(s["anchor"][b0 + m.anchor_join + C.ANCHOR_STAY - 1] - C.BASE)
    return out


def compute_config(key: str, seeds=C.SWEEP_SEEDS) -> dict:
    """Rechnet eine Konfiguration über die gegebenen Läufe: je Kennzahl ein Wert je Lauf und die mittleren Verläufe."""
    import warnings
    group, label, kw = catalog_by_key()[key]
    model = _model(**kw)
    runs = [F.simulate(model, seed) for seed in seeds]
    sums = [summarize(r) for r in runs]
    b0 = model.burn
    series = {name: np.mean([getattr(r, field)[b0:] - (getattr(r, field)[b0 - 1] if base else 0.0) for r in runs], axis=0)
              for name, field, base in (("drift", "bias", True), ("rating", "mean_rating", True), ("level", "mean_true", True), ("slope", "slope", False), ("fit", "fit", True))}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        series["anchor"] = np.nanmean([r.anchor[b0:] for r in runs], axis=0) if model.anchor_join is not None else np.full(model.seasons, NAN)
    return {"key": key, "group": group, "label": label, "runs": {f: [None if x[f] != x[f] else round(float(x[f]), 3) for x in sums] for f in RUN_FIELDS},
            "series": {name: [None if v != v else round(float(v), 3) for v in arr] for name, arr in series.items()}, "seeds": list(seeds)}


def build_all(workers: int = 1) -> dict:
    keys = [row[0] for row in catalog()]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            results = list(pool.map(compute_config, keys))
    else:
        results = [compute_config(k) for k in keys]
    return {"version": 1, "seeds": list(C.SWEEP_SEEDS), "configs": {r["key"]: r for r in results}}


@functools.lru_cache(maxsize=1)
def load_data() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def cfg(key: str) -> dict:
    return load_data()["configs"][key]


def mean(key: str, field: str) -> float:
    xs = np.array([np.nan if x is None else x for x in cfg(key)["runs"][field]], dtype=float)
    return float(np.nanmean(xs)) if np.isfinite(xs).any() else NAN


def se(key: str, field: str) -> float:
    xs = np.array([np.nan if x is None else x for x in cfg(key)["runs"][field]], dtype=float)
    xs = xs[~np.isnan(xs)]
    return float(xs.std(ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else NAN


def sd(key: str, field: str) -> float:
    xs = np.array([np.nan if x is None else x for x in cfg(key)["runs"][field]], dtype=float)
    return float(np.nanstd(xs, ddof=1))


def series(key: str, name: str) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in cfg(key)["series"][name]], dtype=float)


def rows(keys, extra=None) -> list:
    """Zeilen für Tabellen und Abbildungen: Schlüssel, Beschriftung, Drift nach 10/20/40 Saisons mit Standardfehler, Steigung, angepasste Wertung, Bilanzposten."""
    out = []
    for k in keys:
        c = cfg(k)
        r = {"key": k, "label": c["label"], "d10": mean(k, "d10"), "d20": mean(k, "d20"), "d40": mean(k, "d40"), "se": se(k, "d40"), "sd": sd(k, "d40"), "level": mean(k, "level"),
             "rating": mean(k, "rating"), "slope": mean(k, "slope"), "fit": mean(k, "fit"), "series": series(k, "drift")}
        for f in ("entry", "leave", "a", "b", "special", "rest", "rounding", "floor", "uplift", "anchor_in", "cases"):
            r[f] = mean(k, f)
        r["anchor4"], r["anchor4_se"] = mean(k, "anchor4"), se(k, "anchor4")
        r["direct_slope"], r["direct_fit"] = mean(k, "direct_slope"), mean(k, "direct_fit")
        r["entry_err"] = sum(cfg(k)["runs"]["entry_err"]) / max(sum(cfg(k)["runs"]["entry_n"]), 1.0)
        r["leave_err"] = sum(cfg(k)["runs"]["leave_err"]) / max(sum(cfg(k)["runs"]["leave_n"]), 1.0)
        if extra:
            r.update(extra(k))
        out.append(r)
    return out


def created(r: dict) -> float:
    """Punkte, die die Regeln im Spiel erzeugt haben (a, b, Sonderwertung, Untergrenze), Mittel über die Läufe."""
    return r["a"] + r["b"] + r["special"] + r["floor"]


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def verdict_analysis(a: Analysis):
    s = a.settings
    end = float(a.drift[-1])
    text = (f"Nach {s.seasons} Saisons liegt das Wertungsmittel der Gewerteten {signed(a.rating[-1])} Punkte gegenüber dem Ende der Einlaufzeit, die mittlere Fähigkeit {signed(a.level[-1])}: "
            f"**Drift {signed(end)} Punkte**.")
    if np.isfinite(a.anchor).any():
        last = int(np.nonzero(np.isfinite(a.anchor))[0][-1])
        text += f" Der Anker (konstante Stärke 1500) steht bei {signed(a.anchor[last] - C.BASE)} Punkten um 1500."
    if abs(end) < 2 * C.NOISE:
        return "info", text + f" Das liegt im Rauschen eines einzelnen Laufs (rund ±{de(C.NOISE)} Punkte); die Experimente unten mitteln über 60 Läufe."
    if end > 0:
        return "warning", text + " **Inflation:** Die Wertungen liegen über dem, was die Fähigkeit hergibt."
    return "warning", text + " **Deflation:** Die Wertungen bleiben hinter der Fähigkeit zurück."


def verdict_remedies(rs):
    r = {x["key"]: x for x in rs}
    return "warning", (f"Im **geschlossenen Feld** (niemand geht, niemand kommt, alle Stärken fest) bleibt die Wertung ohne Gegenmittel stehen ({signed(r['c_none']['d40'])} Punkte nach {C.EXP_SEASONS} Saisons). "
                       f"Mit allen Regeln der Wertungsordnung steigt sie um **{signed(r['c_all']['d40'])}** (Mittel über 60 Läufe, Fehler ±{de(r['c_all']['se'])}): der Erfolgsaufschlag a allein "
                       f"{signed(r['c_a']['d40'])}, die Sonderwertung 300 allein {signed(r['c_special']['d40'])}, der Bremsfaktor b {signed(r['c_b']['d40'])}, die Untergrenze {signed(r['c_floor']['d40'])}; "
                       f"das ungleiche K0 nach Index und Alter allein {signed(r['c_bare']['d40'])}. Die Punkte kommen aus keiner Gegenbuchung: Sie entstehen im Spiel.")


def verdict_youth(rs):
    r = {x["key"]: x for x in rs}
    return "warning", (f"Kommen die Neuen {C.PRESETS['Lernende Jugend, DWZ-Regeln']['gap']} Punkte unter ihrer späteren Stärke und lernen, **fällt** die Skala ohne Gegenmittel um "
                       f"**{signed(r['y_none']['d40'])}** Punkte (Deflation). Mit allen Regeln der WO **steigt** sie um **{signed(r['y_all']['d40'])}**: die Gegenmittel schießen über. "
                       f"Einzeln, ausgehend von „nur K aus Index, Alter, Wertung“ ({signed(r['y_bare']['d40'])}): mit Erfolgsaufschlag a {signed(r['y_a']['d40'])}, mit Erst-DWZ-Korrektur r {signed(r['y_r']['d40'])}, "
                       f"mit Sonderwertung {signed(r['y_special']['d40'])}; alle Regeln außer a ergeben {signed(r['y_noa']['d40'])}. Der Fehler jedes Mittelwerts liegt bei etwa ±{de(r['y_all']['se'])}.")


def verdict_balance(rs):
    r = {x["key"]: x for x in rs}
    w, n = r["s_weak_none"], C.EXP_N
    return "info", (f"Die Bilanz der Wertungssumme (Punkte je Spieler, Mittel über 60 Läufe) hat drei Quellen: **Eintritt** (die Erst-DWZ der Neuen), **Abgang** (die Wertungen der Gehenden) und das **Spiel**. "
                    f"Das Spiel zahlt nur aus, was Regeln vorsehen: im geschlossenen Feld {signed(created(r['s_closed_all']) / n)} Punkte je Spieler mit allen Regeln, {signed(created(r['s_closed_none']) / n)} ohne Gegenmittel. "
                    f"Ohne Gegenmittel bleibt der Rest dem Austausch: Gehen die Schwächsten, liegt die Erst-DWZ der Neuen im Mittel {signed(w['entry_err'])} Punkte über ihrer Stärke, die Wertung der Gehenden "
                    f"{signed(w['leave_err'])}; je Saison kommen etwa Abgangsquote mal Differenz dazu (0,15 · {de(w['entry_err'] - w['leave_err'])} ≈ {de(C.EXP_CHURN * (w['entry_err'] - w['leave_err']), 1)} Punkte), "
                    f"nach {C.EXP_SEASONS} Saisons gemessen {signed(w['d40'])}. Eine rückstellende Kraft gibt es nicht.")


def verdict_special(rs):
    r = {x["key"]: x for x in rs}
    return "warning", (f"Die **Sonderwertung 300** (WO 3.6.3) ist allein ein großer Punkterzeuger: Im geschlossenen Feld {signed(r['c_special']['d40'])} gegen {signed(r['c_bare']['d40'])} ohne sie "
                       f"({de(r['c_all']['cases'])} Auslösungen je Lauf). Wenn das ganze Feld lernt, dreht sie ohne sonstige Gegenmittel das Vorzeichen: {signed(r['s_learns_none']['d40'])} "
                       f"ohne Gegenmittel gegen {signed(r['s_learns_all']['d40'])} mit allen Regeln. Der Grund: Die Gegner eines Ausnahmespielers rechnen mit seiner höheren Leistung und verlieren deshalb weniger, "
                       "der Ausnahmespieler gewinnt unverändert.")


def verdict_entry(rs):
    r = {x["key"]: x for x in rs}
    return "info", (f"Die Hürde der Erst-DWZ (mindestens 5 Partien und 0,5 Punkte gegen Gewertete) macht aus dem Schätzer eine Auswahl der Glücklichen. Bei zufälligem Abgang liegt die Erst-DWZ im Mittel "
                    f"{signed(r['s_random_none']['entry_err'])} Punkte über der Stärke (ohne Gegenmittel), wenn die Schwächsten gehen {signed(r['s_weak_none']['entry_err'])}; die Drift nach {C.EXP_SEASONS} Saisons "
                    f"ist {signed(r['s_random_none']['d40'])} bzw. {signed(r['s_weak_none']['d40'])}: Gehen die Schwächsten, liegt das Feld höher, die Neuen liegen weiter unter ihm, und der Fehler wächst.")


def verdict_uplift(rs):
    r = {x["key"]: x for x in rs}
    s_on = series("u_random_on", "slope")
    u = C.EXP_UPLIFT
    return "info", (f"Die einmalige Anhebung (Anhang 4) staucht die Skala: Die Steigung der Wertung auf die Stärke liegt vor ihr bei {de(float(series('u_random_off', 'slope')[u - 1]), 2)}, "
                    f"unmittelbar danach bei {de(r['u_random_on']['direct_slope'], 2)}, am Ende der Saison bei {de(float(s_on[u]), 2)} und fünf Saisons später wieder bei {de(float(s_on[u + 5]), 2)}: "
                    f"Die Gewinnerwartung bleibt bei σ = 283, die Abstände "
                    f"dehnen sich zurück. Das Niveau bleibt höher: bei Stärke 1500 {signed(float(series('u_random_on', 'fit')[-1] - series('u_random_off', 'fit')[-1]))} Punkte gegen den Lauf ohne Anhebung. "
                    f"Das Modell prüft nicht, ob 0,64 sachgerecht ist.")


def verdict_anchor(rs):
    r = {x["key"]: x for x in rs}
    first, last = C.ANCHOR_ERAS[0], C.ANCHOR_ERAS[-1]
    return "info", (f"Derselbe Spieler (Stärke konstant), Beitritt nach {first} oder {last} Saisons, Wertung nach {C.ANCHOR_STAY} Saisons: im Feld, das **lernt**, "
                    f"**{signed(r[f'a_learns_{first}']['anchor4'])} → {signed(r[f'a_learns_{last}']['anchor4'])}**, bei der lernenden Jugend {signed(r[f'a_youth_{first}']['anchor4'])} → "
                    f"{signed(r[f'a_youth_{last}']['anchor4'])}, im geschlossenen Feld {signed(r[f'a_closed_{first}']['anchor4'])} → {signed(r[f'a_closed_{last}']['anchor4'])} "
                    f"(Fehler jedes Mittels etwa ±{de(max(r[k]['anchor4_se'] for k in r))}). Der Anker liest die Skalenlage ab, mit Rauschen: Unterschiede unter etwa 20 Punkten sind nicht zu deuten.")


def verdict_noeffect(rs):
    r = {x["key"]: x for x in rs}
    gap = max(abs(r[f"n_{k}_{v}"]["d40"] - r[f"n_{k}_base"]["d40"]) for k in ("closed", "weak") for v in ("table", "even"))
    return "info", (f"Die gedruckte Tabelle P(D) (D ganzzahlig, p auf drei Stellen) und das Rundungsverfahren von Rn ändern die Drift nicht messbar: im geschlossenen Feld {signed(r['n_closed_base']['d40'])} "
                    f"gegen {signed(r['n_closed_table']['d40'])} und {signed(r['n_closed_even']['d40'])}, wenn die Schwächsten gehen {signed(r['n_weak_base']['d40'])} gegen "
                    f"{signed(r['n_weak_table']['d40'])} und {signed(r['n_weak_even']['d40'])} (größter Unterschied {de(gap)}, Fehler etwa ±{de(r['n_weak_base']['se'])}).")


def verdict_variants(rs):
    r = {x["key"]: x for x in rs}
    return "info", (f"Teilnahme nach Aktivität mit Mannschaftsturnier und eine Altersverteilung verschieben die Größen, nicht die Richtung: lernende Jugend {signed(r['v_youth_base']['d40'])} im Grundfall, "
                    f"{signed(r['v_youth_sched']['d40'])} mit Aktivität, {signed(r['v_youth_ages']['d40'])} mit Altersverteilung, {signed(r['v_youth_both']['d40'])} mit beidem; Schwächste gehen "
                    f"{signed(r['v_weak_base']['d40'])}, {signed(r['v_weak_sched']['d40'])}, {signed(r['v_weak_ages']['d40'])}, {signed(r['v_weak_both']['d40'])}.")
