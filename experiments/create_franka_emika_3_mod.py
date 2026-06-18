import numpy as np
import uaibot as ub
from uaibot.utils import Utils

from uaibot.graphics.meshmaterial import MeshMaterial
from uaibot.graphics.model3d import Model3D

from uaibot import Box
# from uaibot.simobjects.ball import Ball
from uaibot.simobjects.box import Box
from uaibot.simobjects.cylinder import Cylinder

from uaibot.robot.links import Link
import uaibot_cpp_bind as ub_cpp


def create_franka_emika_3_mod(
    htm=np.identity(4), name="", color="silver", opacity=1, eef_frame_visible=True
):
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
        [0.00000, 0.0000, 0.0000, 0.00000, 0.00000, 0.0000, 0.0000],
        [
            0.33300,
            0.0000,
            0.3160,
            0.00000,
            0.38400,
            0.0000,
            0.1070 + 0.07,
        ],  # "d" translation in z
        [
            -1.5708,
            1.5708,
            1.5708,
            -1.5708,
            1.57080,
            1.5708,
            0.0000,
        ],  # "alfa" rotation in x
        [
            0.00000,
            0.0000,
            0.0825,
            -0.0825,
            0.00000,
            0.0880,
            0.0000,
        ],  # "a" translation in x 0.25
        [0, 0, 0, 0, 0, 0, 0],
    ]

    n = 7
    scale = 1

    # Collision model
    col_model = [[], [], [], [], [], [], []]

    factor = 0.75
    factor5 = 0.85
    factor4 = 0.78
    factor3 = 0.78
    factor2 = 0.79
    factor1 = 0.7
    col_model[0].append(
        Box(
            htm=Utils.trn([0, 0, 0]),
            name=name + "_C0_0",
            width=0.08 * 2 * factor1,
            depth=0.08 * 2 * factor1,
            height=0.28,
            color="red",
            opacity=0.3,
        )
    )

    col_model[1].append(
        Box(
            htm=Utils.trn([0, 0, 0.18]),
            name=name + "_C1_0",
            width=0.07 * 2 * factor2,
            depth=0.07 * 2 * factor2,
            height=0.25,
            color="blue",
            opacity=0.3,
        )
    )

    col_model[2].append(
        Box(
            htm=Utils.trn([0, 0, 0]),
            name=name + "_C2_0",
            width=0.07 * 2 * factor3,
            depth=0.07 * 2 * factor3,
            height=0.24,
            color="green",
            opacity=0.3,
        )
    )

    col_model[3].append(
        Box(
            htm=Utils.trn([0, 0, 0.13]),
            name=name + "_C3_0",
            width=0.07 * 2 * factor4,
            depth=0.07 * 2 * factor4,
            height=0.20,
            color="yellow",
            opacity=0.3,
        )
    )

    A = np.matrix([[1, 0, 0, 0], [0, 0, 1, -0.383], [0, -1, 0, 0], [0, 0, 0, 1]])

    col_model[4].append(
        Box(
            htm=A * Utils.trn([0, 0.09, 0.28]) * Utils.rotx(-0.24),
            name=name + "_C4_0",
            width=0.08,
            depth=0.05,
            height=0.23,
            color="magenta",
            opacity=0.3,
        )
    )

    col_model[4].append(
        Box(
            htm=Utils.trn([0, 0, -0.03]),
            name=name + "_C4_1",
            width=0.06 * 2 * factor5,
            depth=0.06 * 2 * factor5,
            height=0.22 * 0.9,
            color="magenta",
            opacity=0.3,
        )
    )

    col_model[5].append(
        Box(
            htm=Utils.trn([0, 0, 0.02]),
            name=name + "_C5_0",
            width=0.06 * 2 * factor,
            depth=0.06 * 2 * factor,
            height=0.21,
            color="cyan",
            opacity=0.3,
        )
    )

    col_model[6].append(
        Box(
            htm=Utils.trn([0, 0, 0.04 - 0.09]) * Utils.rotz(-np.pi / 4),
            name=name + "_C6_0",
            width=0.05,
            depth=0.21,
            height=0.10,
            color="red",
            opacity=0.3,
        )
    )

    col_model[6].append(
        Box(
            htm=Utils.trn([0.03, 0.03, -0.025 - 0.07]),
            name=name + "_C6_1",
            width=0.1000,
            depth=0.09,
            height=0.0340,
            color="red",
            opacity=0.3,
        )
    )

    # Create 3d objects
    htm1 = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0008, -1.0, 0.0],
            [0.0, 1.0, 0.0008, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm2 = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0008, 1.0, 0.0],
            [0.0, -1.0, 0.0008, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm3 = np.matrix(
        [
            [1.0, 0.0, 0.0, -0.0825],
            [0.0, 0.0008, 1.0, 0.0],
            [0.0, -1.0, 0.0008, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm4 = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0825],
            [0.0, 0.0008, -1.0, 0.0],
            [0.0, 1.0, 0.0008, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm5 = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0008, 1.0, 0.0],
            [0.0, -1.0, 0.0008, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm6 = np.matrix(
        [
            [1.0, 0.0, 0.0, -0.088],
            [0.0, 0.0008, 1.0, 0.0],
            [0.0, -1.0, 0.0008, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    htm7 = np.matrix(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.177],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    link_3d_obj = []

    base_3d_obj = [
        Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link0.obj',
            scale,
            np.identity(4),
            MeshMaterial(
                metalness=0.3,
                clearcoat=1,
                roughness=0.5,
                normal_scale=[0.5, 0.5],
                color=color,
                opacity=opacity,
            ),
        )
    ]

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link1.obj",
                scale,
                htm1,
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link2.obj",
                scale,
                htm2,
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link3.obj",
                scale,
                htm3,
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link4.obj",
                scale,
                htm4,
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link5.obj",
                scale,
                htm5,
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link6.obj",
                scale,
                htm6,
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            )
        ]
    )

    link_3d_obj.append(
        [
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link7.obj",
                scale,
                htm7,
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            ),
            Model3D(
                "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/hand.obj",
                scale,
                Utils.trn([0, 0, -0.107 + 0.05 - 0.01])
                * Utils.rotx(0)
                * Utils.rotz(-np.pi / 4),  # 0.333+0.295
                MeshMaterial(
                    metalness=0.3,
                    clearcoat=1,
                    roughness=0.5,
                    normal_scale=[0.5, 0.5],
                    color=color,
                    opacity=opacity,
                ),
            ),
        ]
    )

    # Create links

    links = []
    for i in range(n):
        links.append(
            Link(
                i,
                link_info[0][i],
                link_info[1][i],
                link_info[2][i],
                link_info[3][i],
                link_info[4][i],
                link_3d_obj[i],
            )
        )

        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    q0 = [0.0, 0.0, 0.0, -np.pi * 4 / 180, 0.0, 0.0, 0.0]

    # Create joint limits
    joint_limits = (np.pi / 180) * np.matrix(
        [
            [-166, 166],
            [-101, 101],
            [-166, 166],
            [-176, -4],
            [-166, 166],
            [-1, 215],
            [-166, 166],
        ]
    )

    return ub.Robot(
        name,
        links,
        base_3d_obj,
        htm,
        np.identity(4),
        Utils.rotz(-np.pi / 4),
        q0,
        eef_frame_visible,
        joint_limits,
    )

#%% 
# Test the function
from uaibot import Simulation
import webbrowser
from pathlib import Path

def open_in_browser(filename: str):
    """
    Opens an HTML file in the system's default web browser.
    Works cross-platform (Linux, macOS, Windows).
    """
    path = Path(filename).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Convert to file:// URL and open
    webbrowser.open_new_tab(path.as_uri())


if __name__ == "__main__":
    sim = Simulation.create_sim_grid()
    robot = create_franka_emika_3_mod(name="franka_emika_3_mod", color="silver", opacity=1)
# jaco = ub.Robot.create_kinova_gen3()
# res = jaco.compute_dist(box)
# res.jac_dist_mat
    sim.add(robot)

    htm0 = Utils.trn([0.0, 0, 0.8])
    box = Box(htm=htm0, width=0.2, depth=0.2, height=0.2, color="orange", opacity=0.5)
    sim.add(box)
    dsro = robot.signed_distance(box, gamma=2, is_conservative=False, epsilon=1e-3)
    # Each element is the signed distance between i-th link collision primitive and the box.
    signed_distances = dsro.dist_vect
    gradients = dsro.jac_dist_mat
    print("Signed Distances:\n", np.array(signed_distances))
    print("Analytical Gradients:\n", np.array(gradients))
    dsro.dist_vect

    for link in robot.links:
        for col_obj, _ in link.col_objects:
            sim.add(col_obj)

    filename = "franka_emika_3_mod"
    path = "/tmp"
    sim.save(path, f"{filename}")
    open_in_browser(f"{path}/{filename}.html")

