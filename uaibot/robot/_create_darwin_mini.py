from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.ball import *
from uaibot.simobjects.box import *
from uaibot.simobjects.cylinder import *
from uaibot.simobjects.rigidobject import *
from uaibot.simobjects.group import *

from .links import *

from uaibot.robot._create_darwin_mini_arm import _create_darwin_mini_arm
from uaibot.robot._create_darwin_mini_leg_left import _create_darwin_mini_leg_left
from uaibot.robot._create_darwin_mini_leg_right import _create_darwin_mini_leg_right




def _create_darwin_mini(htm, name, color, opacity):

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    param_arm_left = _create_darwin_mini_arm(np.identity(4), "arm_left", color, 1)
    param_arm_right = _create_darwin_mini_arm(np.identity(4), "arm_right", color, 1)
    param_leg_left = _create_darwin_mini_leg_left(np.identity(4), "leg_left", color, 1)
    param_leg_right = _create_darwin_mini_leg_right(np.identity(4), "leg_right", color, 1)

    chestpart1 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/darwin_chest.obj',
        0.004,
        htm * Utils.trn([-0.3, -0.179, 0.37]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                     opacity=opacity, side="DoubleSide"))
    chestpart2 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/frm_body_01.obj',
        0.004,
        htm * Utils.trn([-0.3, -0.179, 0.37]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                     opacity=opacity, side="DoubleSide"))
    chestpart3 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320.obj',
        0.004,
        htm * Utils.trn([0.265, 0.405, 0.32]) * Utils.rotz(-3.14 / 2) * Utils.rotx(3.14 / 3 + 0.25) * Utils.rotz(
            3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=1, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                     opacity=opacity, side="DoubleSide"))

    chestpart4 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320.obj',
        0.004,
        htm * Utils.trn([0.265, -0.405, 1.305]) * Utils.rotx(3.14) * Utils.rotz(-3.14 / 2) * Utils.rotx(
            3.14 / 3 + 0.25) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=1, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                     opacity=opacity, side="DoubleSide"))

    chestpart5 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPU-2.obj',
        0.004,
        htm * Utils.trn([0.05, 0.13, -0.18]) * Utils.rotz(3.14 / 2) * Utils.trn([-0.42, -0.179, 0.575]) * Utils.rotz(
            3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                     opacity=opacity, side="DoubleSide"))

    chestpart6 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320-2.obj',
        0.004,
        htm * Utils.trn([0.69, 0.1, 0.475]) * Utils.rotx(0.08) * Utils.roty(-3.14 / 2) * Utils.trn(
            [-0.3, -0.179, 0.15]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                     opacity=opacity, side="DoubleSide"))

    chestpart7 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPU-2.obj',
        0.004,
        htm * Utils.trn([0.05, -0.06, -0.18]) * Utils.rotz(3.14 / 2) * Utils.trn([-0.42, -0.179, 0.575]) * Utils.rotz(
            3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                     opacity=opacity, side="DoubleSide"))

    chestpart8 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320-2.obj',
        0.004,
        htm * Utils.trn([0.69, -0.09, 0.475]) * Utils.rotx(0.08) * Utils.roty(-3.14 / 2) * Utils.trn(
            [-0.3, -0.179, 0.15]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                     opacity=opacity, side="DoubleSide"))

    chest = RigidObject(
        [chestpart1, chestpart2, chestpart3, chestpart4, chestpart5, chestpart6, chestpart7, chestpart8],
        name + "_chest")

    headpart1 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/darwin_head.obj',
        0.004,
        htm * Utils.trn([-0.3, -0.179, 0.37]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                     opacity=opacity, side="DoubleSide"))

    headpart2 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/eye.obj',
        0.0009,
        htm * Utils.trn([0.177, -0.043, 0.982]) * Utils.rotz(3.14 / 2 - 0.3),
        MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="blue",
                     opacity=opacity, side="DoubleSide"))
    headpart3 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/eye.obj',
        0.0009,
        htm * Utils.trn([0.171, 0.053, 0.982]) * Utils.rotz(3.14 / 2 + 0.4),
        MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="blue",
                     opacity=opacity, side="DoubleSide"))
    headpart4 = Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320-2.obj',
        0.004,
        htm * Utils.trn([-0.3, -0.179, 0.37]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
        MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                     opacity=opacity, side="DoubleSide"))

    head = RigidObject([headpart1, headpart2, headpart3, headpart4], name + "_head")

    return param_arm_left, param_arm_right, param_leg_left, param_leg_right, head, chest
