import numpy as np
import uaibot.robot as rb
from uaibot.utils import Utils
from uaibot.simobjects.cylinder import Cylinder
from uaibot.simobjects.box import Box
from uaibot.graphics.model3d import Model3D
from uaibot.graphics.meshmaterial import MeshMaterial
from uaibot.robot.links import Link
from uaibot.utils import Utils

# TODO
"""The function _create_kinova_gen3 needs to be updated in the future. Its
initial configuration should be set to q0=0, however uaibot does not support
offsets in DHT yet. The initial configuration is set to q0=pi for all joints
after the first joint. This is a workaround and will raise errors if one checks
joint limits, as this initial configuration is outside the joint limits.
"""

def _create_kinova_gen3(
    htm, name="", color="#e6e1e1", opacity=1.0, eef_frame_visible=True
):
    """Model: https://www.kinovarobotics.com/resources Gen3 CAD model (7DoF) [p. 230]
    Parameters: https://www.kinovarobotics.com/resources GEN3 User Guide
    """
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception(
            "The parameter 'htm' should be a 4x4 homogeneous transformation matrix."
        )

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number."
        )

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    link_info = [
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],  # "theta" rotation in z
        [
            -(0.1564 + 0.1284),
            -(0.0054 + 0.0064),
            -(0.2104 + 0.2104),
            -(0.0064 + 0.0064),
            -(0.2084 + 0.1059),
            -(0.0),
            -(0.1059 + 0.0615),
        ],  # "d" translation in z
        [
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi / 2,
            np.pi,
        ],  # "alfa" rotation in x
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],  # "a" translation in x
        [0, 0, 0, 0, 0, 0, 0],
    ]  # joint type

    n = 7

    # Define initial configuration
    q0 = [0.0, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]

    # Create 3d objects
    H00 = Utils.rotx(np.pi)
    H01 = (
        H00
        @ Utils.rotz(q0[0])
        @ Utils.trn([0, 0, link_info[1][0]])
        @ Utils.rotx(link_info[2][0])
        @ Utils.trn([link_info[3][0], 0, 0])
    )
    H02 = (
        H01
        @ Utils.rotz(q0[1])
        @ Utils.trn([0, 0, link_info[1][1]])
        @ Utils.rotx(link_info[2][1])
        @ Utils.trn([link_info[3][1], 0, 0])
    )
    H03 = (
        H02
        @ Utils.rotz(q0[1])
        @ Utils.trn([0, 0, link_info[1][2]])
        @ Utils.rotx(link_info[2][2])
        @ Utils.trn([link_info[3][2], 0, 0])
    )
    H04 = (
        H03
        @ Utils.rotz(q0[1])
        @ Utils.trn([0, 0, link_info[1][3]])
        @ Utils.rotx(link_info[2][3])
        @ Utils.trn([link_info[3][3], 0, 0])
    )
    H05 = (
        H04
        @ Utils.rotz(q0[1])
        @ Utils.trn([0, 0, link_info[1][4]])
        @ Utils.rotx(link_info[2][4])
        @ Utils.trn([link_info[3][4], 0, 0])
    )
    H06 = (
        H05
        @ Utils.rotz(q0[1])
        @ Utils.trn([0, 0, link_info[1][5]])
        @ Utils.rotx(link_info[2][5])
        @ Utils.trn([link_info[3][5], 0, 0])
    )
    H07 = (
        H06
        @ Utils.rotz(q0[1])
        @ Utils.trn([0, 0, link_info[1][6]])
        @ Utils.rotx(link_info[2][6])
        @ Utils.trn([link_info[3][6], 0, 0])
    )
    htm0 = Utils.rotx(np.pi)
    htm_base = np.eye(4)
    htm1 = htm0 @ Utils.rotx(-link_info[2][0]) @ Utils.trn([0, 0, link_info[1][0]])
    htm1 = Utils.inv_htm(H01)
    htm2 = Utils.inv_htm(H02)
    htm3 = Utils.inv_htm(H03)
    htm4 = Utils.inv_htm(H04)
    htm5 = Utils.inv_htm(H05)
    htm6 = Utils.inv_htm(H06)
    htm7 = Utils.inv_htm(H07)

    mesh1 = MeshMaterial(
        metalness=0.7,
        clearcoat=1,
        roughness=0.5,
        normal_scale=[0.5, 0.5],
        color=color,
        opacity=opacity,
    )
    scale = 1

    base_3d_obj = [
        Model3D(
            "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_base.obj",
            scale,
            htm_base,
            mesh1,
        )
    ]

    link_3d_obj = []

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_shoulder.obj",
                scale,
                htm1,
                mesh1,
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_halfarm1.obj",
                scale,
                htm2,
                mesh1,
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_halfarm2.obj",
                scale,
                htm3,
                mesh1,
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_forearm.obj",
                scale,
                htm4,
                mesh1,
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_wrist1.obj",
                scale,
                htm5,
                mesh1,
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_wrist2.obj",
                scale,
                htm6,
                mesh1,
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KinovaGen3/Kinova7dof_eefNcamera.obj",
                scale,
                htm7,
                mesh1,
            )
        ]
    )

    ## Inertial Parameters
    # Base mass: 1.697 kg
    list_mass = [1.377, 1.1636, 1.1636, 0.930, 0.678, 0.678, 0.500]

    # Base COM: [-0.000648, -0.000166, 0.084487]
    list_com = [
        [-0.000023, -0.010364, -0.073360],
        [-0.000044, -0.099580, -0.013278],
        [-0.000044, -0.006641, -0.117892],
        [-0.000018, -0.075478, -0.015006],
        [0.000001, -0.009432, -0.063883],
        [0.000001, -0.045483, -0.009650],
        [-0.000281, -0.011402, -0.029798],
    ]

    def inertia_map(Ixx, Ixy, Ixz, Iyy, Iyz, Izz):
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])

    # Base Inertia Matrix: [[0.004622, 0.000009, 0.000060], [*, 0.004495, 0.000009], [*, *, 0.002079]]
    list_inertia_mat_ = [
        inertia_map(0.004570, 0.000001, 0.000002, 0.004831, 0.000448, 0.001409),
        inertia_map(0.011088, 0.000005, 0.000000, 0.001072, -0.000691, 0.011255),
        inertia_map(0.010932, 0.000000, -0.000007, 0.011127, 0.000606, 0.001043),
        inertia_map(0.008147, -0.000001, 0.000000, 0.000631, -0.000500, 0.008316),
        inertia_map(0.001596, 0.000000, 0.000000, 0.001607, 0.000256, 0.000399),
        inertia_map(0.001641, 0.000000, 0.000000, 0.000410, -0.000278, 0.001641),
        inertia_map(0.000587, 0.000003, 0.000003, 0.000369, 0.000118, 0.000609),
    ]
    # Map inertia matrices do DHT frames using Steiner's theorem:
    # list_inertia_mat = [
    #     M - (list_mass[i] * Utils.S(list_com[i]) @ Utils.S(list_com[i]))
    #     for i, M in enumerate(list_inertia_mat_)
    # ]

    ## Collision models
    col_model = [[], [], [], [], [], [], []]
    # BEST COLLISION PRIMITIVE FOR BASE: CYLINDER
    # -----------------
    # [[ 3.09016994e-01  9.51056516e-01  0.00000000e+00 -4.30947181e-03]
    #  [-9.51056516e-01  3.09016994e-01  0.00000000e+00  2.38524478e-18]
    #  [ 0.00000000e+00  0.00000000e+00  1.00000000e+00  8.54240000e-02]
    #  [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
    #  radius = 0.0507586109592414
    #  height = 0.170848
    opacity_col = 0.5
    htm_col1 = Utils.inv_htm(H01) @  np.array(
        [
            [3.09016994e-01, 9.51056516e-01, 0.00000000e00, -7.07423947e-06],
            [-9.51056516e-01, 3.09016994e-01, 0.00000000e00, 6.92558177e-03],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 2.44797000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[0].append(
        Cylinder(
            htm=htm_col1,
            name=name + "_C0_0",
            radius=0.05299158224551782,
            height=0.1719979999999999,
            color="red",
            opacity=opacity_col,
        )
    )

    htm_col2 = Utils.inv_htm(H02) @  np.array(
        [
            [-6.65636469e-04, -9.99999776e-01, -7.46406454e-05, 6.83465448e-06],
            [-9.93771408e-01, 6.69808282e-04, -1.11435810e-01, -2.19731881e-02],
            [1.11435835e-01, 1.34681170e-20, -9.93771631e-01, 3.76340258e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[1].append(
        Cylinder(
            htm=htm_col2,
            name=name + "_C1_0",
            radius=0.05033478624779629,
            height=0.2772392440068092,
            color="green",
            opacity=opacity_col,
        )
    )

    htm_col3 = Utils.inv_htm(H03) @  np.array(
        [
            [-3.09016994e-01, 9.51056516e-01, -3.78436673e-17, -3.25325421e-18],
            [9.51056516e-01, 3.09016994e-01, 1.16470832e-16, -7.86769618e-03],
            [1.22464680e-16, 0.00000000e00, -1.00000000e00, 6.24542000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[2].append(
        Cylinder(
            htm=htm_col3,
            name=name + "_C2_0",
            radius=0.051909320764667,
            height=0.253996,
            color="blue",
            opacity=opacity_col,
        )
    )

    htm_col4 = Utils.inv_htm(H04) @  np.array(
        [
            [-1.00000000e00, 0.00000000e00, -1.22464680e-16, -1.99496634e-06],
            [0.00000000e00, 1.00000000e00, 0.00000000e00, -3.71880150e-02],
            [1.22464680e-16, 0.00000000e00, -1.00000000e00, 7.92989500e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[3].append(
        Cylinder(
            htm=htm_col4,
            name=name + "_C3_0",
            radius=0.047732015109487665,
            height=0.26689699999999994,
            color="red",
            opacity=opacity_col,
        )
    )

    htm_col5 = Utils.inv_htm(H05) @   np.array(
        [
            [8.09016994e-01, 5.87785252e-01, 0.00000000e00, 7.29375607e-08],
            [-5.87785252e-01, 8.09016994e-01, 0.00000000e00, -1.49007990e-02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 9.85708000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[4].append(
        Cylinder(
            htm=htm_col5,
            name=name + "_C4_0",
            radius=0.04459820102528319,
            height=0.13849,
            color="green",
            opacity=opacity_col,
        )
    )

    htm_col6 = Utils.inv_htm(H06) @  np.array(
        [
            [1.41128252e-04, 9.99999989e-01, -4.35454376e-05, -2.08329652e-06],
            [-9.55547739e-01, 1.47693563e-04, 2.94836391e-01, -3.82612302e-02],
            [2.94836394e-01, 0.00000000e00, 9.55547749e-01, 1.06598120e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[5].append(
        Cylinder(
            htm=htm_col6,
            name=name + "_C5_0",
            radius=0.04176774959071834,
            height=0.16636604407989197,
            color="blue",
            opacity=opacity_col,
        )
    )

    htm_col7_0 = Utils.inv_htm(H07) @  np.array(
        [
            [8.09016994e-01, 5.87785252e-01, 0.00000000e00, 4.00000000e-06],
            [-5.87785252e-01, 8.09016994e-01, 0.00000000e00, -1.67555410e-02],
            [0.00000000e00, 0.00000000e00, 1.00000000e00, 1.15927300e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[6].append(
        Cylinder(
            htm=htm_col7_0,
            name=name + "_C6_0",
            radius=0.0456054591322633,
            height=0.06192399999999987,
            color="red",
            opacity=opacity_col,
        )
    )
    htm_col7_1 = Utils.inv_htm(H07) @  np.array(
        [
            [0.00000000e00, -2.62118281e-04, 9.99999966e-01, 1.34355361e-06],
            [0.00000000e00, 9.99999966e-01, 2.62118281e-04, 3.39492827e-02],
            [-1.00000000e00, 0.00000000e00, 0.00000000e00, 1.17558950e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    col_model[6].append(
        Box(
            htm=htm_col7_1,
            name=name + "_C6_1",
            width=0.021513,
            height=0.09099698,
            depth=0.03175192,
            color="red",
            opacity=opacity_col,
        )
    )

    ## Create links
    links = []
    for i in range(n):
        links.append(
            Link(
                joint_number=i,
                theta=link_info[0][i],
                d=link_info[1][i],
                alpha=link_info[2][i],
                a=link_info[3][i],
                joint_type=link_info[4][i],
                list_model_3d=link_3d_obj[i],
                mass=list_mass[i],
                center_of_mass=list_com[i],
                inertia_tensor=list_inertia_mat_[i],
                wrt_dh=False,
                # com_coordinates=list_com[i], # DEPRECATED PARAMETERS IN UAIBOT>=1.2.2
                # mass=list_mass[i],
                # inertia_matrix=list_inertia_mat[i],
            )
        )

        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    ## Create joint limits
    joint_limits = np.matrix(
        [
            [-np.inf, np.inf],
            [-np.deg2rad(128.9), np.deg2rad(128.9)],
            [-np.inf, np.inf],
            [-np.deg2rad(147.8), np.deg2rad(147.8)],
            [-np.inf, np.inf],
            [-np.deg2rad(120.3), np.deg2rad(120.3)],
            [-np.inf, np.inf],
        ]
    )

    htm_eef = np.eye(4)

    robot_ = rb.Robot(
        name,
        links,
        base_3d_obj,
        htm,
        htm0,
        htm_eef,
        q0,
        eef_frame_visible=eef_frame_visible,
        joint_limits=joint_limits,
    )

    return robot_
