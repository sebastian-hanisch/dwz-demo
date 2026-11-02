"""Gemeinsame Fixtures: kleine Modelle und von Hand gebaute Zustände, an denen jede Einheit nachgerechnet wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dwz_field as F  # noqa: E402

ACC_FIELDS = ("entry", "entry_err", "entry_n", "leave", "leave_err", "leave_n", "play_a", "play_b", "play_special", "play_rest", "floor_lift", "rounding", "uplift_gain", "anchor_in",
              "special_cases", "iter_step")


def small(**kw) -> F.Model:
    """Kleines Modell (12 Spieler, 3 Saisons nach 2 Saisons Einlaufzeit), für Einheiten-, Orakel- und Eigenschaftstests."""
    base = {"n": 12, "seasons": 3, "burn": 2, "churn": 0.25}
    base.update(kw)
    return F.Model(**base)


def make_state(model: F.Model, rating, true, age=None, idx=None, rated=None, seed: int = 1):
    """Zustand von Hand: ein Platz je Eintrag von `rating` (n Spieler, der Anker-Platz bleibt inaktiv)."""
    dr = F.draws(seed, model.total, model.slots)
    st = F._State(model, dr)
    n = model.n
    assert len(rating) == n
    st.R[:n] = rating
    st.true[:n] = true
    st.pot[:n] = true
    if age is not None:
        st.age[:n] = age
    if idx is not None:
        st.idx[:n] = idx
    if rated is not None:
        st.rated[:n] = rated
    return st, dr


def fresh_acc() -> dict:
    return {f: 0.0 for f in ACC_FIELDS}


def run_tournament(st, dr, model, games, part=None, now: float = 0.0):
    """Wertet ein Turnier mit fest vorgegebenen Partien aus: `games` ist eine Liste (i, j, Punkte von i); beide Sichten werden ergänzt."""
    P, O, S = [], [], []
    for i, j, w in games:
        P += [i, j]
        O += [j, i]
        S += [w, 1.0 - w]
    original = F._games
    F._games = lambda *_a, **_k: (np.array(P, dtype=int), np.array(O, dtype=int), np.array(S, dtype=float))
    acc = fresh_acc()
    try:
        F._tournament(st, dr, 0, 0, 9, np.ones(model.slots, dtype=bool) if part is None else part, now, acc)
    finally:
        F._games = original
    return acc


@pytest.fixture
def model():
    return small()
