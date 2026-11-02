"""Rechnet alle Experimente über die 60 festen Läufe (SWEEP_SEEDS) und schreibt data/experiments.json.

Die App liest diese Datei, statt die rund 50 Konfigurationen beim Start zu rechnen (je Konfiguration 60 Läufe zu etwa 0,2 s). Der Test `tests/test_experiments_data.py` rechnet
einzelne Läufe jeder Gruppe neu und vergleicht sie mit der Datei; die Datei gehört deshalb ins Repo und muss nach jeder Änderung an Regeln oder Feldmodell neu gebaut werden.

    python tools/build_experiments.py [Prozesse]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import dwz_evaluation as ev  # noqa: E402


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    data = ev.build_all(workers)
    ev.DATA_FILE.parent.mkdir(exist_ok=True)
    ev.DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(data['configs'])} Konfigurationen, {ev.DATA_FILE.stat().st_size / 1024:.0f} KB -> {ev.DATA_FILE}")


if __name__ == "__main__":
    main()
