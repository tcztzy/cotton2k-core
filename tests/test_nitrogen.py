import numpy as np

from cotton2k.core.nitrogen import PlantNitrogen
from cotton2k.core.phenology import Stage


def test_nitrogen_allocation():
    pn = PlantNitrogen()
    pn.carbon_allocated_for_root_growth = 1
    pn.extra_carbon = 0
    pn.total_actual_leaf_growth = 0.1
    pn.total_actual_petiole_growth = 0.1
    pn.actual_stem_growth = 0.1
    pn.actual_square_growth = 0.1
    pn.actual_burr_growth = 0.1
    pn.actual_boll_growth = 0.1
    pn.fruiting_nodes_stage = np.zeros((3, 30, 5), dtype=np.int_)
    pn.fruiting_nodes_stage[0, 0, 0] = Stage.GreenBoll
    pn.fruiting_nodes_boll_weight = np.zeros((3, 30, 5))
    pn.fruiting_nodes_boll_weight[0, 0, 0] = 0.2
    pn.npool = 0.03
    pn.leaf_nitrogen = 0
    pn.petiole_nitrogen = 0
    pn.stem_nitrogen = 0
    pn.root_nitrogen = 0
    pn.square_nitrogen = 0
    pn.seed_nitrogen = 0
    pn.burr_nitrogen = 0
    pn.nitrogen_allocation()
    assert pn.reqtot == 0.038504
    assert pn.npool == 0
    assert pn.seed_nitrogen == 0.005183999999999999
