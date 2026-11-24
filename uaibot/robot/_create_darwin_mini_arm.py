from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.ball import *
from uaibot.simobjects.box import *
from uaibot.simobjects.cylinder import *

from .links import *


def _create_darwin_mini_arm(htm, name, color, opacity):
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    #[np.pi / 2, -np.pi / 90, np.pi / 90]
    link_info = [[0, 0, 0],  # "theta" rotation in z
                 [0.08, 0, 0],  # "d" translation in z
                 [np.pi / 2, 0, 0],  # "alfa" rotation in x
                 [0.05, 0.18, 0.23],  # "a" translation in x
                 [0, 0, 0]]  # joint type

    scale = 1
    n = 3

    opacity = 1

    col_model = [[], [], []]
    # Create 3d objects

    Q01 = Utils.rotz(np.pi/2) * Utils.rotz(link_info[0][0]) * Utils.trn([0, 0, link_info[1][0]]) * Utils.rotx(link_info[2][0]) * Utils.trn(
        [link_info[3][0], 0, 0])
    Q02 = Utils.rotz(-np.pi / 90) * Utils.rotz(link_info[0][1]) * Utils.trn([0, 0, link_info[1][1]]) * Utils.rotx(link_info[2][1]) * Utils.trn(
        [link_info[3][1], 0, 0])
    Q03 = Utils.rotz(np.pi / 90) * Utils.rotz(link_info[0][2]) * Utils.trn([0, 0, link_info[1][2]]) * Utils.rotx(link_info[2][2]) * Utils.trn(
        [link_info[3][2], 0, 0])

    base_3d_obj = []

    link_3d_obj = []

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/darwin_ext_01.obj',
            0.004,
            Utils.inv_htm(Q01)  * Utils.trn([-0.315, 0.435, -0.325]) * Utils.rotz(-3.14) * Utils.roty(-3.14 / 2),
            MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity, side="DoubleSide"))]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320.obj',
            0.004,
            Utils.inv_htm(Q01 * Q02) * Utils.trn([0.004, 0.01, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
            MeshMaterial(metalness=1, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                         opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPO.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02) * Utils.trn([0.004, 0.01, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/frm_ext_02.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02) * Utils.trn([0.004, 0.01, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide"))
        ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/XL-320.obj',
            0.004,
            Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.19, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
            MeshMaterial(metalness=1, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color="#302b2b",
                         opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPO.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.19, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPU.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.013, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPD_2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.0136, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPO.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.roty(-np.pi / 2) * Utils.rotx(-np.pi / 2) * Utils.trn(
                    [-0.325, -0.385, -0.0115]),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPD_1.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.0136, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPL-2b2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.0136, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPL-2b2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([-0.044, 0.0136, -0.02]) * Utils.rotx(
                    -3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPL-2b2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.46, 0.502, 0.075]) * Utils.rotx(3.14 / 2) * Utils.rotz(
                    3.14 / 2) * Utils.trn([-0.044, 0.0136, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPL-2b2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0, 0, 0.083]) * Utils.roty(3.14) * Utils.trn(
                    [0.46, 0.502, 0.075]) * Utils.rotx(3.14 / 2) * Utils.rotz(
                    3.14 / 2) * Utils.trn([-0.044, 0.0136, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPD_1.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.035, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPD_3.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.004, 0.035, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPL-2b2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.007, 0.905, 0.043]) * Utils.rotx(3.14) * Utils.roty(
                    -3.14 / 2) * Utils.trn([0, 0, 0.083]) * Utils.roty(3.14) * Utils.trn(
                    [0.46, 0.502, 0.075]) * Utils.rotx(3.14 / 2) * Utils.rotz(
                    3.14 / 2) * Utils.trn([-0.044, 0.0136, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
                MeshMaterial(metalness=0.5, clearcoat=0, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                             opacity=opacity, side="DoubleSide")),
            Model3D(
                'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/DarwinMini/SPL-2b2.obj',
                0.004,
                Utils.inv_htm(Q01 * Q02 * Q03) * Utils.trn([0.055, 0.905, 0.043]) * Utils.rotx(3.14) * Utils.roty(
                    -3.14 / 2) * Utils.trn([0, 0, 0.083]) * Utils.roty(3.14) * Utils.trn(
                    [0.46, 0.502, 0.075]) * Utils.rotx(3.14 / 2) * Utils.rotz(
                    3.14 / 2) * Utils.trn([-0.044, 0.0136, -0.02]) * Utils.rotx(-3.14 / 12) * Utils.trn(
                    [0, 0.2, 0]) * Utils.rotz(3.14) * Utils.trn([0.318, -0.2, -0.3]) * Utils.roty(-3.14 / 2),
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
    q0 = [0, 0, 0]

    # Create joint limits
    joint_limits = (np.pi / 180) * np.matrix([[-180, 180], [-180, 180], [-180, 180]])

    ###
    # Get all variables in the global scope
    # for i in range(12):
    #     try:
    #         matrix_str = str(eval('htm'+str(i+1)))
    #         matrix_str = matrix_str.strip()

    #         # Replace multiple spaces with a single space
    #         matrix_str = "\n".join(" ".join(row.split()) for row in matrix_str.split("\n"))

    #         # Split into rows, remove extra brackets
    #         rows = matrix_str.strip("[]").split("\n")

    #         # Format each row to insert commas between numbers
    #         formatted_rows = []
    #         for row in rows:
    #             row = row.strip()
    #             elements = row.strip("[]").split()
    #             formatted_row = "[" + ", ".join(elements) + "]"
    #             formatted_rows.append(formatted_row)

    #         # Join rows in a single line
    #         formatted_matrix = "[" + ", ".join(formatted_rows) + "]"
    #         print("htm"+str(i+1)+" = np.matrix("+formatted_matrix+")")
    #     except:
    #         print("")
    ###
    
    return base_3d_obj, links, np.identity(4), np.identity(4), q0, joint_limits
