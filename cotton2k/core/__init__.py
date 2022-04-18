"""Cotton2k model."""
from pathlib import Path

from .simulation import Simulation

__all__ = ("run",)


def run(profile_path: "Path | str | dict") -> Simulation:
    return Simulation(profile_path).run()
