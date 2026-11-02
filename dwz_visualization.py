"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import dwz_constants as C
import dwz_evaluation as ev


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _uplift_line(fig, a):
    if a.settings.uplift_at is not None:
        fig.add_vline(x=a.settings.uplift_at + 1, line_dash="dot", line_color="rgba(150,150,150,0.9)", annotation_text="Anhebung", annotation_position="top left")


def build_overview(a: ev.Analysis):
    """Wertungsmittel und mittlere Fähigkeit der Gewerteten (Punkte, DWZ-Skala) je Saison, der Anker als eigene Linie. Die Lücke ist die Drift."""
    x = np.arange(1, len(a.rating) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=C.BASE + a.rating, mode="lines", name="Wertungsmittel der Gewerteten", line=dict(color=C.COLOR_RATING, width=3),
                             hovertemplate="Saison %{x}: Wertungsmittel %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=C.BASE + a.level, mode="lines", name="mittlere Fähigkeit derselben (nur in der Simulation bekannt)",
                             line=dict(color=C.COLOR_TRUTH, width=2, dash="dot"), hovertemplate="Saison %{x}: Fähigkeit %{y:.0f}<extra></extra>"))
    if np.isfinite(a.anchor).any():
        fig.add_trace(go.Scatter(x=x, y=a.anchor, mode="lines", name="Anker (konstante Stärke 1500)", line=dict(color=C.COLOR_ANCHOR, width=1.8), connectgaps=False,
                                 hovertemplate="Saison %{x}: Anker %{y:.0f}<extra></extra>"))
    _uplift_line(fig, a)
    fig.update_xaxes(title_text="Saison nach der Einlaufzeit")
    fig.update_yaxes(title_text="Punkte (DWZ-Skala)")
    return _base(fig, height=400)


def build_scale(a: ev.Analysis):
    """Steigung der Wertung auf die wahre Stärke (1 = der Maßstab stimmt) je Saison."""
    x = np.arange(1, len(a.slope) + 1)
    fig = go.Figure()
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    fig.add_trace(go.Scatter(x=x, y=a.slope, mode="lines", name="Steigung Wertung auf Stärke", line=dict(color=C.COLOR_RATING, width=3),
                             hovertemplate="Saison %{x}: Steigung %{y:.2f}<extra></extra>"))
    _uplift_line(fig, a)
    fig.update_xaxes(title_text="Saison nach der Einlaufzeit")
    fig.update_yaxes(title_text="Steigung (1 = Maßstab stimmt)")
    return _base(fig, height=280)


def build_drift(a: ev.Analysis):
    """Drift je Saison (Wertungsmittel minus Fähigkeit)."""
    x = np.arange(1, len(a.drift) + 1)
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    fig.add_trace(go.Scatter(x=x, y=a.drift, mode="lines", name="Drift (Wertung − Fähigkeit)", line=dict(color=C.COLOR_RATING, width=3),
                             hovertemplate="Saison %{x}: Drift %{y:.0f} Punkte<extra></extra>"))
    _uplift_line(fig, a)
    fig.update_xaxes(title_text="Saison nach der Einlaufzeit")
    fig.update_yaxes(title_text="Punkte")
    return _base(fig, height=300)


BALANCE_LABELS = {"net": "Eintritt − Abgang", "a": "Erfolgsaufschlag a", "b": "Bremsfaktor b", "special": "Sonderwertung 300", "rest": "ungleiches K0, Rundung, Untergrenze",
                  "uplift": "Anhebung", "anchor": "Anker (Beitritt)"}


def balance_posts(b: dict) -> dict:
    """Die sieben Posten der Bilanz (Summe der Wertungen, Punkte): Eintritt minus Abgang, a, b, Sonderwertung, ungleiches K0 mit Rundung und Untergrenze, Anhebung, Anker."""
    return {"net": b["entry"] - b["leave"], "a": b["a"], "b": b["b"], "special": b["special"], "rest": b["rest"] + b["rounding"] + b["floor"], "uplift": b["uplift"], "anchor": b["anchor"]}


def build_balance(a: ev.Analysis):
    """Bilanz der Wertungssumme der Gewerteten (Punkte je Feldspieler): was sie vom Ende der Einlaufzeit auf den Endwert gebracht hat."""
    posts = balance_posts(a.bal)
    n = a.run.model.n
    values = [posts[k] / n for k in posts]
    names = [BALANCE_LABELS[k] for k in posts]
    fig = go.Figure(go.Waterfall(x=names + ["Änderung der Summe"], y=values + [0], measure=["relative"] * len(values) + ["total"], text=[ev.signed(v) for v in values] + [ev.signed(a.bal["end"] / n)],
                                 textposition="outside", increasing=dict(marker=dict(color=C.COLOR_CREATED)), decreasing=dict(marker=dict(color=C.COLOR_RATING)),
                                 totals=dict(marker=dict(color=C.COLOR_ANCHOR)), hovertemplate="%{x}: %{y:.0f} Punkte je Spieler<extra></extra>"))
    fig.update_yaxes(title_text="Punkte je Feldspieler")
    return _base(fig, height=340)


def build_drift_lines(rows, keys=None):
    """Mittlere Drift (60 Läufe) je Saison für mehrere Konfigurationen."""
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    sel = [r for r in rows if keys is None or r["key"] in keys]
    for k, r in enumerate(sel):
        x = np.arange(1, len(r["series"]) + 1)
        fig.add_trace(go.Scatter(x=x, y=r["series"], mode="lines", name=r["label"], line=dict(color=C.PALETTE[k % len(C.PALETTE)], width=2.5),
                                 hovertemplate=r["label"] + ", Saison %{x}: %{y:.0f} Punkte<extra></extra>"))
    fig.update_xaxes(title_text="Saison nach der Einlaufzeit")
    fig.update_yaxes(title_text="Drift in Punkten (Mittel über 60 Läufe)")
    return _base(fig, height=360)


def build_bars(rows, field="d40", title="Drift nach 40 Saisons (Punkte)"):
    """Ein Balken je Konfiguration (Mittel über 60 Läufe) mit Standardfehler."""
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=[r[field] for r in rows], error_y=dict(type="data", array=[r.get("se", 0.0) for r in rows]),
                           marker_color=[C.PALETTE[k % len(C.PALETTE)] for k in range(len(rows))], text=[ev.signed(r[field]) for r in rows], textposition="outside",
                           hovertemplate="%{x}: %{y:.0f}<extra></extra>"))
    fig.update_yaxes(title_text=title)
    return _base(fig)


def build_stack(rows):
    """Bilanz je Konfiguration (Punkte je Feldspieler, Mittel über 60 Läufe), gestapelt nach Posten."""
    fig = go.Figure()
    n = C.EXP_N
    for k, post in enumerate(BALANCE_LABELS):
        ys = []
        for r in rows:
            posts = balance_posts({"entry": r["entry"], "leave": r["leave"], "a": r["a"], "b": r["b"], "special": r["special"], "rest": r["rest"], "rounding": r["rounding"], "floor": r["floor"],
                                   "uplift": r["uplift"], "anchor": r["anchor_in"]})
            ys.append(posts[post] / n)
        fig.add_trace(go.Bar(x=[r["label"] for r in rows], y=ys, name=BALANCE_LABELS[post], marker_color=C.PALETTE[k % len(C.PALETTE)], hovertemplate=BALANCE_LABELS[post] + ": %{y:.0f}<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="Punkte je Feldspieler (Summe der Wertungen)")
    return _base(fig, height=420)


def build_slopes(keys, labels=None):
    """Mittlere Steigung der Wertung auf die Stärke (60 Läufe) je Saison; die Anhebung ist senkrecht markiert."""
    fig = go.Figure()
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(150,150,150,0.8)")
    for k, key in enumerate(keys):
        y = ev.series(key, "slope")
        fig.add_trace(go.Scatter(x=np.arange(1, len(y) + 1), y=y, mode="lines", name=ev.cfg(key)["label"], line=dict(color=C.PALETTE[k % len(C.PALETTE)], width=2.5),
                                 hovertemplate=ev.cfg(key)["label"] + ", Saison %{x}: %{y:.2f}<extra></extra>"))
    fig.add_vline(x=C.EXP_UPLIFT + 1, line_dash="dot", line_color="rgba(150,150,150,0.9)", annotation_text="Anhebung", annotation_position="top left")
    fig.update_xaxes(title_text="Saison nach der Einlaufzeit")
    fig.update_yaxes(title_text="Steigung (1 = Maßstab stimmt)")
    return _base(fig, height=340)


def build_anchor(rs):
    """Wertung des Ankers (Punkte um 1500) nach 4 Saisons je Szenario und Beitrittszeit; Fehlerbalken: Standardfehler."""
    shades = ("#9cc5bf", "#5f9d95", "#2F6B65")
    scen = [("closed", "Geschlossenes Feld"), ("weak", "Die Schwächsten gehen"), ("youth", "Lernende Jugend"), ("learns", "Das Feld lernt")]
    by = {r["key"]: r for r in rs}
    fig = go.Figure()
    for k, era in enumerate(C.ANCHOR_ERAS):
        fig.add_trace(go.Bar(x=[label for _, label in scen], y=[by[f"a_{key}_{era}"]["anchor4"] for key, _ in scen], error_y=dict(type="data", array=[by[f"a_{key}_{era}"]["anchor4_se"] for key, _ in scen]),
                             name=f"Beitritt nach {era} Saisons", marker_color=shades[k], hovertemplate="%{x}, Beitritt nach " + str(era) + ": %{y:.0f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Wertung des Ankers minus 1500 (Punkte)")
    return _base(fig)
