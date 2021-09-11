import pytest

from cotton2k.core import read_input
from cotton2k.core.simulation import Simulation


def test_read_input(empty_json, test_json):
    with pytest.raises((KeyError, TypeError)):
        read_input(empty_json)
    with pytest.raises((KeyError, TypeError)):
        read_input({})
    with pytest.raises((KeyError, TypeError)):
        read_input(str(empty_json))
    read_input(test_json)


def test_write_output(sim: Simulation):
    assert len(sim.states) == 181
    state = sim.states[-1]
    assert (state.lint_yield - 2219.4412479701537) < 1e-9
    assert (state.leaf_weight - 6.384953819149585) < 1e-9
    assert (state.root_weight - 54.748594488769854) < 1e-9
    assert (state.stem_weight - 27.388671543526538) < 1e-9
    assert (state.plant_weight - 170.88176149303297) < 1e-9
    assert "date" in state.keys()
