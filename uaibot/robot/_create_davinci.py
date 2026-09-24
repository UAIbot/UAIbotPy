import uaibot.robot as rb
from uaibot.utils import Utils

from uaibot.simobjects.group import Group
from uaibot.robot._create_davinci_arms import _create_davinci_arm1, _create_davinci_arm2, _create_davinci_arm3, _create_davinci_arm4
from uaibot.robot._create_davinci_chest import _create_davinci_chest
import numpy as np


def _create_davinci(htm, name, color='#3e3f42', opacity=1, eef_frame_visible=True, joint_limits=None):

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception(
            "The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception(
            "The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception(
            "The parameter 'opacity' should be a float between 0 and 1.")

    arm1_links, arm1_base_3d_obj, arm1_htmbase, arm1_htmeef, arm1_q0, arm1_limits = _create_davinci_arm1(
        color=color, opacity=opacity, name=name + '_arm_1')
    arm2_links, arm2_base_3d_obj, arm2_htmbase, arm2_htmeef, arm2_q0, arm2_limits = _create_davinci_arm2(
        color=color, opacity=opacity, name=name + '_arm_2')
    arm3_links, arm3_base_3d_obj, arm3_htmbase, arm3_htmeef, arm3_q0, arm3_limits = _create_davinci_arm3(
        color=color, opacity=opacity, name=name + '_arm_3')
    arm4_links, arm4_base_3d_obj, arm4_htmbase, arm4_htmeef, arm4_q0, arm4_limits = _create_davinci_arm4(
        color=color, opacity=opacity, name=name + '_arm_4')
    chest = _create_davinci_chest(name=name, color=color, opacity=opacity)

    robot_arm1 = rb.Robot(name + "__arm1", links=arm1_links, list_base_3d_obj=arm1_base_3d_obj, htm=np.identity(4),
                          htm_base_0=arm1_htmbase, htm_n_eef=arm1_htmeef, q0=arm1_q0, eef_frame_visible=eef_frame_visible, joint_limits=arm1_limits)
    robot_arm2 = rb.Robot(name + "__arm2", links=arm2_links, list_base_3d_obj=arm2_base_3d_obj, htm=np.identity(4),
                          htm_base_0=arm2_htmbase, htm_n_eef=arm2_htmeef, q0=arm2_q0, eef_frame_visible=eef_frame_visible, joint_limits=arm2_limits)
    robot_arm3 = rb.Robot(name + "__arm3", links=arm3_links, list_base_3d_obj=arm3_base_3d_obj, htm=np.identity(4),
                          htm_base_0=arm3_htmbase, htm_n_eef=arm3_htmeef, q0=arm3_q0, eef_frame_visible=eef_frame_visible, joint_limits=arm3_limits)
    robot_arm4 = rb.Robot(name + "__arm4", links=arm4_links, list_base_3d_obj=arm4_base_3d_obj, htm=np.identity(4),
                          htm_base_0=arm4_htmbase, htm_n_eef=arm4_htmeef, q0=arm4_q0, eef_frame_visible=eef_frame_visible, joint_limits=arm4_limits)
    return Group([robot_arm1, robot_arm2, robot_arm3, robot_arm4, chest])
