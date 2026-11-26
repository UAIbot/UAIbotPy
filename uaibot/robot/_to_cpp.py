import uaibot_cpp_bind as cpp
from uaibot.utils import obj_to_cpp


class _CPPManipulator(cpp.CPP_Manipulator):
    """Kinematics and collision interface with an optional native plant."""

    dynamics = None

    @property
    def state(self):
        return None if self.dynamics is None else self.dynamics.state

    @property
    def parameters(self):
        return None if self.dynamics is None else self.dynamics.parameters


def _set_joint_parameters(cpp_robot, links, limits):
    for i, link in enumerate(links):
        cpp_robot.set_joint_param(
            i,
            link.theta,
            link.d,
            link.alpha,
            link.a,
            int(link.joint_type),
            limits[i, 0],
            limits[i, 1],
        )


def _to_cpp(robot):
    links = robot.links
    cpp_robot = _CPPManipulator(len(links))
    cpp_robot.set_htm_extra(robot.htm_base_0, robot.htm_n_eef)
    _set_joint_parameters(cpp_robot, links, robot.joint_limit)
    for i, link in enumerate(links):
        for obj, local_pose in link.col_objects:
            cpp_robot.add_geo_prim(i, obj_to_cpp(obj, local_pose))
    return cpp_robot
