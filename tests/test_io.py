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
    for i, s in enumerate(sim.states):
        if 0 <= i <= 14:
            assert i >= 0 and s.pre_fruiting_leaf_area == [0] * 9
        if 15 <= i <= 16:
            assert s.pre_fruiting_leaf_area == [0, 0.04, 0, 0, 0, 0, 0, 0, 0]
        elif i == 17:
            assert s.pre_fruiting_leaf_area[1] - 0.04854027307818367 < 1e-9
        elif i == 18:
            assert s.pre_fruiting_leaf_area[1] - 0.05691214973504026 < 1e-9
        elif i == 19:
            assert s.pre_fruiting_leaf_area[1] - 0.06742381894057693 < 1e-9
        elif i == 20:
            assert s.pre_fruiting_leaf_area[1] - 0.07354398298576931 < 1e-9
        elif i == 21:
            assert s.pre_fruiting_leaf_area[1] - 0.07856394039688848 < 1e-9
    assert (state.lint_yield - 2219.4412479701537) < 1e-9
    assert (state.ginning_percent - 0.37750407781082385) < 1e-9
    assert (state.leaf_weight - 6.416714304185261) < 1e-9
    assert (state.root_weight - 57.962083610964996) < 1e-9
    assert (state.stem_weight - 28.14266502833103) < 1e-9
    assert (state.plant_weight - 174.06201458313524) < 1e-9
    assert (state.open_bolls_weight - 51.13404240829509) < 1e-9
    assert (state.open_bolls_burr_weight - 0) < 1e-9
    assert (state.green_bolls_weight - 0) < 1e-9
    assert (state.green_bolls_burr_weight - 0) < 1e-9
    assert (state.petiole_weight - 0.055159545577716476) < 1e-9
    assert (state.square_weight - 0) < 1e-9
    assert (state.reserve_carbohydrate - 0) < 1e-9
    assert (state.number_of_open_bolls - 10.66628490996115) < 1e-9
    assert (state.number_of_green_bolls - 10.654268246224818) < 1e-9
    assert (state.number_of_squares - 1) < 1e-9
    assert "date" in state.keys()
