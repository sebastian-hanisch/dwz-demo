"""PDF-Export: erzeugt für jede Konfiguration gültige PDF-Bytes; der Latin-1-Filter ersetzt alle Zeichen außerhalb der Kernschrift."""

import dwz_constants as C
import dwz_evaluation as ev
import dwz_pdf_export as X
from dwz_scenario import Settings


def test_pdf_is_built_for_every_mechanism():
    for s in (Settings(n=20, seasons=6), Settings(n=20, seasons=6, kmode=C.KMODE[1], use_a=False, use_b=False, special=False, use_r=False, use_floor=False),
              Settings(n=20, seasons=6, use_uplift=True, uplift=2, gap=0, trend=3.5, sched=C.SCHED[1], ages=True), Settings(n=20, seasons=6, churn=0.0, use_anchor=False),
              Settings(n=20, seasons=6, leave=C.LEAVE[1], gap=200)):
        data = X.build_pdf(ev.analyse(s))
        assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_filter_replaces_greek_and_math_symbols():
    out = X._latin("τ ≈ 0,5; σ → φ µ θ − **fett** € ± Σ")
    assert out == "Tau ca. 0,5; sigma -> phi mu theta - fett EUR +- Summe"
    assert out.encode("latin-1")
