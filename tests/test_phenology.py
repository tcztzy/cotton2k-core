from dataclasses import dataclass
from datetime import date

import numpy as np

from cotton2k.core.phenology import Phenology


def pre_fruiting_node_leaf_abscission(droplf, first_square_date, defoliate_date):
    ...


@dataclass
class node_leaf:
    ...


@dataclass
class node:
    leaf: node_leaf


@dataclass
class fb:
    number_of_fruiting_nodes: int
    nodes: list[node]


@dataclass
class vb:
    number_of_fruiting_branches: int
    fruiting_branches: list[fb]


def test_phenology():
    phenology = Phenology()
    phenology.leaf_area_index = 0.0001
    assert phenology.leaf_abscission(4, date(2020, 7, 1), date(2020, 9, 1)) is None
    phenology.leaf_area_index = 1
    phenology.reserve_carbohydrate = 0
    phenology.version = 0x400
    phenology.pre_fruiting_node_leaf_abscission = pre_fruiting_node_leaf_abscission
    phenology.number_of_vegetative_branches = 1
    phenology.vegetative_branches = [
        vb(5, [fb(1, [node(node_leaf())]) for _ in range(5)])
    ]
    phenology.date = date(2020, 8, 1)
    phenology.leaf_area = 4
    phenology.node_leaf_age = np.zeros((3, 30, 5), dtype=np.double)
    phenology.node_leaf_age[0, :5, 0] = 1
    phenology.leaf_abscission(4, date(2020, 7, 1), date(2020, 9, 1))
