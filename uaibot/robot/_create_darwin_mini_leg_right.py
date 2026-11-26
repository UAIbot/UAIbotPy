from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.ball import *
from uaibot.simobjects.box import *
from uaibot.simobjects.cylinder import *

from .links import *


def _create_darwin_mini_leg_right(htm, name, color, opacity):
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    link_info = [[0, 0, 0, 0],  # "theta" rotation in z
                 [0.055, 0, 0, -0.125],  # "d" translation in z
                 [np.pi / 2, 0, np.pi / 2, -np.pi / 2],  # "alfa" rotation in x
                 [0.035, 0.185, 0.171, 0],  # "a" translation in x
                 [0, 0, 0, 0]]  # joint type


    n = 4
    col_model = [[], [], [], []]

    # Create 3d objects

    Q01 = Utils.rotz(np.pi / 2) * Utils.rotz(link_info[0][0]) * Utils.trn([0, 0, link_info[1][0]]) * Utils.rotx(link_info[2][0]) * Utils.trn(
        [link_info[3][0], 0, 0])
    Q02 = Utils.rotz(link_info[0][1]) * Utils.trn([0, 0, link_info[1][1]]) * Utils.rotx(link_info[2][1]) * Utils.trn(
        [link_info[3][1], 0, 0])
    Q03 = Utils.rotz(link_info[0][2]) * Utils.trn([0, 0, link_info[1][2]]) * Utils.rotx(link_info[2][2]) * Utils.trn(
        [link_info[3][2], 0, 0])
    Q04 = Utils.rotz(np.pi / 2) * Utils.rotz(link_info[0][3]) * Utils.trn([0, 0, link_info[1][3]]) * Utils.rotx(link_info[2][3]) * Utils.trn(
        [link_info[3][3], 0, 0])

    base_3d_obj = []

    link_3d_obj = []

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPD_1.obj',
            0.004,
            Utils.inv_htm(Q01) * Utils.trn([-0.487, -0.315, -0.06]) * Utils.rotx(-np.pi / 2) * Utils.rotz(-0.3),
            MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320-2.obj',
                0.004,
                Utils.inv_htm(Q01) * Utils.trn([0.04, 0.73, 0.155]) * Utils.rotx(np.pi / 2) * Utils.trn(
                    [0.69, -0.09, 0.655]) * Utils.rotx(0.08) * Utils.roty(-3.14 / 2) * Utils.trn(
                    [-0.3, -0.179, 0.15]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
                MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                             opacity=opacity, side="DoubleSide"))
        ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/frm_ext_02.obj',
            0.004,
            Utils.inv_htm(Q01 * Q02) * Utils.trn([0.317, -0.01 - 0.035, -0.435]) * Utils.rotx(0.275) * Utils.roty(
                -np.pi / 2),
            MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPO.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02) * Utils.trn([0.317, -0.35, -0.345]) * Utils.roty(-np.pi / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPO.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02) * Utils.trn([0.04, -0.01 + 0.925, 0.155]) * Utils.rotx(np.pi / 2) * Utils.trn(
                    [0.69, -0.09, 0.655]) * Utils.rotx(0.08) * Utils.roty(-3.14 / 2) * Utils.trn(
                    [-0.3, -0.179, 0.15]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
                MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                             opacity=opacity, side="DoubleSide"))
        ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/mini_cr_calf.obj',
            0.004,
            Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([-0.271, 0.445, -0.325]),
            MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPO.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.317, -0.35 + 0.185, -0.345]) * Utils.roty(-np.pi / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320-2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02) * Utils.trn([0.04, -0.01 + 0.925, 0.155]) * Utils.rotx(np.pi / 2) * Utils.trn(
                    [0.69, -0.09, 0.655]) * Utils.rotx(0.08) * Utils.roty(-3.14 / 2) * Utils.trn(
                    [-0.3, -0.179, 0.15]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
                MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPO.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02) * Utils.trn([0.317, -0.35 + 0.171, -0.345]) * Utils.roty(-np.pi / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide"))
        ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320-2.obj',
            0.004,
            Utils.inv_htm(Q01 * Q02 * Q03 * Q04) * Utils.trn([0, 0.785, 0]) * Utils.rotz(np.pi) * Utils.trn(
                [0.3, 0.295, -0.062]) * Utils.rotz(np.pi / 2) * Utils.roty(-np.pi / 2) * Utils.trn(
                [0.04, 1, 0]) * Utils.rotx(np.pi / 2) * Utils.trn(
                [0.69, -0.09, 0.655]) * Utils.rotx(0.08) * Utils.roty(-3.14 / 2) * Utils.trn(
                [-0.3, -0.179, 0.15]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14 / 2),
            MeshMaterial(metalness=0.8, clearcoat=0.5, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                         opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPD_1.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03 * Q04) * Utils.trn([-0.487, 0.05, -0.06]) * Utils.rotx(
                    -np.pi / 2) * Utils.rotz(-0.3),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/foot.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03 * Q04) * Utils.trn([-0.06 + 0.34, 0.19, -0.31]) * Utils.rotz(np.pi),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/sole.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03 * Q04) * Utils.trn([-0.06 + 0.34, 0.19, -0.31]) * Utils.rotz(np.pi),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide"))
        ]
    )

    # Create links

    links = []
    for i in range(n):
        links.append(Link(i, link_info[0][i], link_info[1][i], link_info[2][i], link_info[3][i], link_info[4][i],
                          link_3d_obj[i]))

        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    q0 = [0.000, 0.000, 0.000, 0.000]

    # Create joint limits
    joint_limits = (np.pi / 180) * np.matrix([[-180, 180], [-180, 180], [-180, 180], [-180, 180]])

    return base_3d_obj, links, np.identity(4), Utils.trn([-0.037,-0.025,-0.129]), q0, joint_limits
