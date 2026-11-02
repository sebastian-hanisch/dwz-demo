"""Szenario: Reglerwerte (Settings) und das daraus erzeugte Modell des offenen Feldes."""

from dataclasses import dataclass

import dwz_constants as C
import dwz_field as F


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seasons: int = C.DEFAULT_SEASONS
    churn: float = C.DEFAULT_CHURN
    leave: str = C.DEFAULT_LEAVE
    gap: int = C.DEFAULT_GAP
    rate: float = C.DEFAULT_RATE
    trend: float = C.DEFAULT_TREND
    kmode: str = C.KMODE[0]
    kfix: int = C.DEFAULT_KFIX                 # nur bei festem K
    use_a: bool = True
    use_b: bool = True
    use_r: bool = True
    use_floor: bool = True
    special: bool = True
    sched: str = C.SCHED[0]
    ages: bool = False
    use_uplift: bool = False
    uplift: int = C.DEFAULT_UPLIFT            # nur mit Anhebung
    use_anchor: bool = True
    join: int = C.DEFAULT_JOIN                # nur mit Anker
    seed: int = C.DEFAULT_SEED

    @property
    def uplift_at(self):
        """Saison (nach der Einlaufzeit), zu deren Beginn die Anhebung gilt (None = keine)."""
        if not self.use_uplift:
            return None
        return min(max(self.uplift, C.UPLIFT_MIN), uplift_max(self.seasons))

    @property
    def anchor_join(self):
        """Saison (nach der Einlaufzeit), zu deren Beginn der Anker beitritt (None = kein Anker); er spielt mindestens eine Saison."""
        if not self.use_anchor:
            return None
        return min(max(self.join, C.JOIN_MIN), join_max(self.seasons))


def join_max(seasons: int) -> int:
    """Spätester Beitritt des Ankers: danach bleibt mindestens eine Saison."""
    return seasons - 2


def uplift_max(seasons: int) -> int:
    """Späteste Saison der Anhebung (zu Beginn der letzten Saison gilt sie noch)."""
    return seasons - 1


def make_model(s: Settings) -> F.Model:
    return F.Model(n=s.n, seasons=s.seasons, churn=s.churn, leave=s.leave, gap=float(s.gap), rate=s.rate, trend=s.trend, kmode=s.kmode, kfix=float(s.kfix), use_a=s.use_a,
                   use_b=s.use_b, use_r=s.use_r, use_floor=s.use_floor, special=s.special, sched=s.sched, ages=s.ages, uplift_at=s.uplift_at, anchor_join=s.anchor_join)
