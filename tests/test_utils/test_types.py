import pytest
from uaibot.simobjects import Ball, Box, Cylinder, ConvexPolytope, PointCloud, Frame
from uaibot.robot import Robot
from uaibot.utils.types import Simple, Metric, Groupable, ObjSim

@pytest.mark.parametrize("factory", [Ball, Box, Cylinder, ConvexPolytope])
def test_simple_is_metric_and_groupable(factory):
    obj = factory()
    assert isinstance(obj, Simple)
    assert isinstance(obj, Metric)
    assert isinstance(obj, Groupable)
    assert isinstance(obj, ObjSim)

def test_pointcloud_is_metric_not_simple():
    pc = PointCloud(points=[[0,0,0],[1,0,0],[0,1,0]])
    assert isinstance(pc, Metric)
    assert not isinstance(pc, Simple)
    assert not isinstance(pc, Groupable)
