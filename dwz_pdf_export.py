"""PDF-Export der aktuellen Konfiguration und der Bilanz (fpdf2, Kernschrift: nur Zeichen aus Latin-1)."""

from fpdf import FPDF

import dwz_constants as C
import dwz_evaluation as ev
from dwz_visualization import BALANCE_LABELS, balance_posts


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("φ", "phi"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-"), ("±", "+-"), ("Σ", "Summe")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("DWZ: Gegenmittel gegen Deflation"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(f"Spieler: {s.n}, Saisons nach der Einlaufzeit: {s.seasons}, Abgang {ev.pct(s.churn)} % je Saison ({s.leave}), Seed: {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    rules = [f"K: {s.kmode}" + (f" ({s.kfix})" if s.kmode != C.KMODE[0] else "")]
    for on, name in ((s.use_a, "Erfolgsaufschlag a"), (s.use_b, "Bremsfaktor b"), (s.use_r, "Erst-DWZ-Korrektur r"), (s.use_floor, "Untergrenze 1100"), (s.special, "Sonderwertung 300")):
        rules.append(name + (": ja" if on else ": nein"))
    rules.append(f"Turniere: {s.sched}" + ("; Altersverteilung" if s.ages else ""))
    if s.gap:
        rules.append(f"Neue {s.gap} Punkte unter ihrer Spitze, Aufholen {ev.pct(s.rate)} % je Saison")
    if s.trend:
        rules.append(f"Feld {ev.signed(s.trend, 1)} Punkte je Saison")
    if s.uplift_at is not None:
        rules.append(f"Anhebung 2026 zu Beginn von Saison {s.uplift_at + 1}")
    pdf.multi_cell(0, 5.5, _latin("; ".join(rules) + "."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    posts = balance_posts(a.bal)
    n = a.run.model.n
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, _latin("Bilanz der Wertungssumme (Punkte je Feldspieler)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for key, value in posts.items():
        pdf.cell(100, 6, _latin(BALANCE_LABELS[key]), border=1)
        pdf.cell(40, 6, _latin(ev.signed(value / n, 1)), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.cell(100, 6, _latin("Änderung der Summe"), border=1)
    pdf.cell(40, 6, _latin(ev.signed(a.bal["end"] / n, 1)), border=1, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Spieler", "Alter", "Index", "Wertung", "wahre Stärke", "Abweichung")
    widths = (25, 20, 20, 30, 35, 30)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for row in ev.field_table(a):
        for value, w in zip(row.values(), widths):
            pdf.cell(w, 6, _latin("-" if value is None else str(value)), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (wahre Stärken aus dem Generator der Ranking-Linie, danach Abgang, Zugang und Lernen); die wahre Stärke ist nur in der Simulation bekannt, ihr Nullpunkt (1500) ist "
        "eine Konvention. Regeln nach der Wertungsordnung des Deutschen Schachbundes, gültig ab 8. Juni 2026; keine Aussage über die reale DWZ. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."),
        new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
