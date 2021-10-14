from dataclasses import dataclass
from datetime import date

import numpy as np

from cotton2k.core.phenology import Phenology


def pre_fruiting_node_leaf_abscission(droplf, first_square_date, defoliate_date):
    ...


def test_phenology():
    phenology = Phenology()
    phenology.leaf_area_index = 0.0001
    assert phenology.leaf_abscission(4, date(2020, 7, 1), date(2020, 9, 1)) is None
    phenology.leaf_area_index = 1
    phenology.reserve_carbohydrate = 0
    phenology.version = 0x400
    phenology.pre_fruiting_node_leaf_abscission = pre_fruiting_node_leaf_abscission
    phenology.date = date(2020, 8, 1)
    phenology.leaf_area = 4
    phenology.node_leaf_age = np.zeros((3, 30, 5), dtype=np.double)
    phenology.node_leaf_age[0, :5, 0] = 1
    phenology.main_stem_leaf_weight = np.zeros((3, 30), dtype=np.double)
    phenology.main_stem_leaf_weight[0, :5] = 1
    phenology.node_leaf_weight = np.zeros((3, 30, 5), dtype=np.double)
    phenology.node_leaf_weight[0, :5, 0] = 1
    phenology.leaf_abscission(4, date(2020, 7, 1), date(2020, 9, 1))
