import pytest

from cotton2k.core.simulation import Simulation


def test_read_input(empty_json, test_json):
    with pytest.raises((KeyError, TypeError)):
        Simulation(empty_json)
    with pytest.raises((KeyError, TypeError)):
        Simulation({})
    with pytest.raises((KeyError, TypeError)):
        Simulation(str(empty_json))
    Simulation(test_json)


def test_write_output(sim: Simulation):
    assert len(sim.states) == 181
    state = sim.states[-1]
    assert (state.lint_yield - 2219.4412479701537) < 1e-9
    assert (state.leaf_weight - 6.416714304185261) < 1e-9
    assert (state.root_weight - 54.905901804422214) < 1e-9
    assert (state.stem_weight - 28.14266502833103) < 1e-9
    assert (state.plant_weight - 174.06201458313524) < 1e-9
    assert "date" in state.keys()
