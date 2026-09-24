from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.mtlmeshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.ball import *
from uaibot.simobjects.box import *
from uaibot.simobjects.cylinder import *

from .links import *


def _create_magician_e6(htm, name, color, opacity):
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")


    link_info = [[0, 0, 0, 0, 0, 0], #"theta" rotation in z
                 [0.167, -0.086, 0.086, -0.086, 0.101, -0.065],  # "d" translation in z
                 [-np.pi/2, 0, 0, np.pi/2, -np.pi/2, np.pi],  # "alfa" rotation in x
                 [0, 0.189, 0.16, 0, 0, 0],  # "a" translation in x 0.25
                 [0, 0, 0, 0, 0, 0]] #joint type

    n = 6
    scale = 1

    # Collision model
    col_model = [[], [], [], [], [], []]


    col_model[0].append(Cylinder(htm=Utils.trn([0, 0.05, 0])*Utils.rotx(np.pi/2),
                                 name=name + "_C0_0", radius=0.045, height=0.175, color="red", opacity=0.3))
    col_model[0].append(Cylinder(htm=Utils.trn([0, 0, -0.04]),
                                 name=name + "_C0_1", radius=0.045, height=0.175, color="red", opacity=0.3))
    col_model[1].append(Box(htm=Utils.trn([-0.07, 0, -0.005]),
                            name=name + "_C1_0", width=0.18, depth=0.09, height=0.06, color="blue", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([0, 0, 0.05]),
                                 name=name + "_C1_1", radius=0.045, height=0.175, color="blue", opacity=0.3))
    col_model[2].append(Box(htm=Utils.trn([-0.07, 0, 0.01]),
                            name=name + "_C2_0", width=0.18, depth=0.09, height=0.06, color="green", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([0, 0, -0.04]),
                                 name=name + "_C2_1", radius=0.04, height=0.175, color="green", opacity=0.3))
    col_model[3].append(Cylinder(htm=Utils.trn([0, 0, 0.05]),
                                 name=name + "_C3_0", radius=0.035, height=0.13, color="yellow", opacity=0.3))
    col_model[4].append(Cylinder(htm=Utils.trn([0, 0, 0]),
                                 name=name + "_C3_1", radius=0.035, height=0.14, color="cyan", opacity=0.3))

    # Create 3d objects
    

    htm0 = np.array([[ 1.,  0.,  0.,  0.], [ 0.,  0., -1.,  0.], [ 0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  1.]])
    htm1 = np.array([[0., -0., 1., 0.], [0., -1., -0., 0.0402], [1., 0., 0., 0.], [0., 0., 0., 1.]])
    htm2 = np.array([[0., 0., -1., -0.1892], [1., 0., 0., 0.], [0., -1., -0., 0.04], [0., 0., 0., 1.]])
    htm3 = np.array([[0., 0., -1., -0.1601], [1., 0., 0., 0.], [0., -1., -0., -0.049], [0., 0., 0., 1.]])
    htm4 = np.array([[1., 0., 0., 0.], [0., -1., -0., 0.032], [0., 0., -1., -0.0001], [0., 0., 0., 1.]])
    htm5 = np.array([[0., -0., 1., 0.], [0., -1., -0., 0.0341], [1., 0., 0., 0.], [0., 0., 0., 1.]])
    htm6 = np.array([[1., 0., 0., 0.], [0., 0., -1., -0.0001], [0., 1., 0., -0.018], [0., 0., 0., 1.]])
    
    default_material = MeshMaterial(metalness=0.3, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5],color=color, opacity=opacity)


    base_3d_obj = [Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/base_link.obj',
        scale,
        htm0,
        MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/base_link.mtl',
                         opacity=opacity) if color=='' else default_material)
                   ]

    link_3d_obj = []

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link1.obj',
            scale,
            htm1,
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link1.mtl',
                         opacity=opacity) if color=='' else default_material)
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link2.obj',
            scale,
            htm2,
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link2.mtl',
                         opacity=opacity) if color=='' else default_material)
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link3.obj',
            scale,
            htm3,
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link3.mtl',
                         opacity=opacity) if color=='' else default_material)
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link4.obj',
            scale,
            htm4,
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link4.mtl',
                         opacity=opacity) if color=='' else default_material)
         ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link5.obj',
            scale,
            htm5,
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link5.mtl',
                         opacity=opacity) if color=='' else default_material)
         ]
    )
    
    
    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/MagicianE6/Link6.obj',
            scale,
            htm6,
            mesh_material=MeshMaterial(color = 'silver' ,metalness = 1.0, roughness = 0.0))
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

    htm_base_0 = Utils.trn([0, 0, 0])
    q0 = [-np.pi/2, -np.pi/2, 0.0, np.pi/2, 0.0, 0.0]

    #Create joint limits
    joint_limits = np.matrix([[-2.5*np.pi, 1.5*np.pi],[-np.pi/2-2.356,-np.pi/2+2.356],[-2.6878, 2.6878],[-2.7925+np.pi/2,2.7925+np.pi/2],[-3.0194,3.0194],[-6.28,6.28]])

    return base_3d_obj, links, htm_base_0, np.identity(4), q0, joint_limits
