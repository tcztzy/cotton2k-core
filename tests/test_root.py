import pytest

from cotton2k.core.root import depth_of_layer


def test_depth_of_layer():
    assert depth_of_layer(0) == 2
    with pytest.raises(IndexError):
        depth_of_layer(40)
