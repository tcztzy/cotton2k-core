"""Cotton2k model."""
import csv
import datetime
import json
from importlib.metadata import metadata, version
from pathlib import Path
from typing import TYPE_CHECKING

from _cotton2k import Climate, SoilImpedance, SoilInit

from .simulation import Simulation

if TYPE_CHECKING:
    from email.message import Message
    from typing import Union  # pylint: disable=ungrouped-imports

__all__ = ("run",)
__version__: str = version("cotton2k.core")
meta: "Message" = metadata("cotton2k.core")
__author__: str = meta["Author"]
__license__: str = meta["License"]


SOIL_IMPEDANCE = SoilImpedance()
with open(Path(__file__).parent / "soil_imp.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    SOIL_IMPEDANCE.curves = list(
        map(
            lambda row: {
                (k if k == "water" else float(k)): float(v) for k, v in row.items()
            },
            reader,
        )
    )


def read_input(path: "Union[Path, str, dict]") -> Simulation:
    if isinstance(path, dict):
        kwargs = path
    else:
        kwargs = json.loads(Path(path).read_text())
    sim = Simulation(kwargs.pop("id", 0), kwargs.pop("version", 0x0400), **kwargs)
    soil = SoilInit(**kwargs.pop("soil", {}))  # type: ignore[arg-type]
    start_date = kwargs["start_date"]
    if not isinstance(start_date, (datetime.date, str)):
        raise ValueError
    sim.year = (
        start_date.year
        if isinstance(start_date, datetime.date)
        else int(start_date[:4])
    )
    sim.read_input(lyrsol=soil.lyrsol, **kwargs)
    climate_start_date = kwargs.pop("climate_start_date", 0)
    sim.climate = Climate(climate_start_date, kwargs.pop("climate"))[sim.start_date :]  # type: ignore[misc]  # pylint: disable=line-too-long
    return sim


def run(profile_path: "Union[Path, str, dict]") -> Simulation:
    sim = read_input(profile_path)
    sim.run()
    return sim
