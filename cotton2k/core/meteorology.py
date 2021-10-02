from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import DefaultDict

METEOROLOGY: "DefaultDict" = defaultdict(lambda: defaultdict(dict))
