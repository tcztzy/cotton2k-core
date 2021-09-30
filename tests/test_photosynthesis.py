import datetime

import pytest

from cotton2k.core.photo import Photosynthesis


def test_photosynthesis():
    photo = Photosynthesis()
    photo.date = datetime.date(2020, 10, 1)
    photo.leaf_nitrogen_concentration = 0.07
    photo.day_length = 12
    photo.light_interception = 1
    photo.daytime_temperature = 24
    photo.maintenance_weight = 42
    photo.leaf_area_index = 1
    photo.plant_height = 70
    photo.version = 0x400
    assert (photo.compute_light_interception(1, 100) - 0.75292) < 1e-9
    photo.version = 0x500
    assert (photo.compute_light_interception(1, 100) - 1) < 1e-9
    with pytest.raises(RuntimeError):
        photo.leaf_area_index = 0
        photo.get_net_photosynthesis(0, 0, 0, 0)
    photo.leaf_area_index = 1
    photo.get_net_photosynthesis(0, 0, 0, 0)
