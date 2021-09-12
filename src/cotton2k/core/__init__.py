"""Cotton2k model."""
from importlib.metadata import metadata, version
from pathlib import Path
from typing import TYPE_CHECKING

from .simulation import Simulation

if TYPE_CHECKING:
    from email.message import Message
    from typing import Union  # pylint: disable=ungrouped-imports

__all__ = ("run",)
__version__: str = version("cotton2k.core")
meta: "Message" = metadata("cotton2k.core")
__author__: str = meta["Author"]
__license__: str = meta["License"]


def run(profile_path: "Union[Path, str, dict]") -> Simulation:
    return Simulation(profile_path).run()
