from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.ball import *
from uaibot.simobjects.box import *
from uaibot.simobjects.cylinder import *

from .links import *


def _create_staubli_tx60(htm, name, color, opacity):
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    c = (np.pi / 180)
    link_info = [[0, 0, 0, 0, 0, 0], #"theta" rotation in z
                 [0, 0, 0.02, 0.31, 0, 0.07],  # "d" translation in z
                 [3.14 / 2, 0, 3.14 / 2, 3.14 / 2, -3.14 / 2, 0],  # "alfa" rotation in x
                 [0, 0.29, 0, 0, 0, 0],  # "a" translation in x 0.25
                 [0, 0, 0, 0, 0, 0]] #joint type

    n = 6
    scale = 1

    # Collision model
    col_model = [[], [], [], [], [], []]


    col_model[0].append(Cylinder(htm=Utils.trn([0, -0.15, 0]) * Utils.rotx(-3.14 / 2),
                                 name=name + "_C0_0", radius=0.1, height=0.45, color="red", opacity=0.3))
    col_model[1].append(Box(htm=Utils.trn([-0.15, 0, -0.125]) * Utils.rotz(3.14 / 2) * Utils.rotx(-3.14 / 2 - c * 7.3),
                            name=name + "_C1_0", width=0.18, depth=0.1, height=0.45, color="blue", opacity=0.3))
    col_model[2].append(Box(htm=Utils.trn([0, -0.02, 0]),
                            name=name + "_C2_0", width=0.17, depth=0.21, height=0.25, color="green", opacity=0.3))
    col_model[3].append(Box(htm=Utils.trn([0, -0.08, 0]),
                            name=name + "_C3_0", width=0.15, depth=0.27, height=0.13, color="orange", opacity=0.3))

    # Create 3d objects
    
    htm0 = np.matrix([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    htm1 = np.matrix([[1., 0., 0., 0.], [0., 1., -0.0008, -0.005], [0., 0.0008, 1., 0.], [0., 0., 0., 1.]])
    htm2 = np.matrix([[0.0008, 0.9918, -0.1279, -0.31], [-1., 0.0008, -0.0001, 0.], [0., 0.1279, 0.9918, 0.02], [0., 0., 0., 1.]])
    htm3 = np.matrix([[-1., 0., -0.0016, 0.], [-0.0016, -0., 1., 0.005], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    htm4 = np.matrix([[-1., -0., 0.0016, 0.], [0., 1., -0.0008, -0.31], [-0.0016, -0.0008, -1., 0.], [0., 0., 0., 1.]])
    htm5 = np.matrix([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    htm6 = np.matrix([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    
    
    
    base_3d_obj = [Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/StaubliTX60/base_link.obj',
        scale,
        htm0,
        MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5],
                     color=color, opacity=opacity))]

    link_3d_obj = []

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/StaubliTX60/link1.obj',
            scale,
            htm1,
            MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity))
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/StaubliTX60/link2.obj',
            scale,
            htm2,
            MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity))
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/StaubliTX60/link3.obj',
            scale,
            htm3,
            MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity))
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/StaubliTX60/link4.obj',
            scale,
            htm4,
            MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color=color,
                         opacity=opacity))
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/StaubliTX60/link5.obj',
            scale,
            htm5,
            MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color="#C0C0C0",
                         opacity=opacity))
         ]
    )
    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/StaubliTX60/link6.obj',
            scale,
            htm6,
            MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color="black",
                         opacity=opacity))
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

    htm_base_0 = Utils.trn([0, 0, 0.375])

    q0 = [0, 3.14/2, 0, 0, 0, 0]

    #Create joint limits
    joint_limits = (np.pi/180)*np.matrix([[-180,180],[-127.5+90, 127.5+90],[-142.5+90, 142.5+90],[-270,270],[-121,132.5],[-270,270]])
    
    return base_3d_obj, links, htm_base_0, np.identity(4), q0, joint_limits
