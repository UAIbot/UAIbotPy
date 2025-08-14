from utils import *

from graphics.meshmaterial import *
from graphics.mtlmeshmaterial import *
from graphics.model3d import *

from simobjects.ball import *
from simobjects.box import *
from simobjects.cylinder import *

from .links import *


def _create_kuka_kr5(htm, name, color, opacity):
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
                 [0.335, 0.000, 0.000, -0.405, 0.000, 0.080],  # "d" translation in z
                 [-1.570, 0.000, 1.570, -1.570, -1.570, 0],  # "alfa" rotation in x
                 [0.075, 0.365, 0.090, 0.000, 0.000, 0],  # "a" translation in x
                 [0, 0, 0, 0, 0, 0]] #joint type

    n = 6

    # Collision model
    col_model = [[], [], [], [], [], []]

    col_model[0].append(Cylinder(htm=Utils.trn([-0.075, 0.170, 0]) * Utils.roty(3.14 / 2) * Utils.rotx(3.14 / 2),
                                 name=name + "_C0_0", radius=0.12, height=0.33, color="red", opacity=0.3))

    col_model[0].append(Cylinder(htm=Utils.trn([0, 0, 0.030]) * Utils.rotz(3.14 / 2) * Utils.rotx(3.14),
                                 name=name + "_C0_1", radius=0.095, height=0.30, color="red", opacity=0.3))

    col_model[1].append(Box(htm=Utils.trn([-0.200, 0.020, 0.12]) * Utils.roty(3.14 / 2),
                            name=name + "_C1_0", width=0.1, height=0.5, depth=0.16, color="green", opacity=0.3))

    col_model[1].append(Cylinder(htm=Utils.trn([0, 0, 0.040]) * Utils.rotz(3.14) * Utils.rotx(3.14),
                                 name=name + "_C1_1", radius=0.095, height=0.28, color="green", opacity=0.3))

    col_model[2].append(Box(htm=Utils.trn([0, 0, -0.224]) * Utils.rotz(-3.14 / 2) * Utils.rotx(-3.14 / 2),
                            name=name + "_C2_0", width=0.143, height=0.12, depth=0.45, color="blue", opacity=0.3))

    # Create 3d objects
    htm1 = np.matrix([[-0.0024, 0.0008, -1., 0.], [-1., -0., 0.0024, 0.], [0., 1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm2 = np.matrix([[1., 0., 0., -0.08], [0., -1., -0.0016, 0.135], [0., 0.0016, -1., 0.], [0., 0., 0., 1.]])
    htm3 = np.matrix([[0.24, 0.0008, -0.9708, -0.37], [-0.9708, 0.0002, -0.24, 0.], [0., 1., 0.0008, 0.11], [0., 0., 0., 1.]])
    htm4 = np.matrix([[0.0008, 1., 0., -0.09], [-1., 0.0008, 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    htm5 = np.matrix([[0.0008, -1., 0., 0.], [1., 0.0008, 0., -0.185], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    htm6 = np.matrix([[-1., 0., 0.0016, 0.], [0., 1., 0., 0.], [-0.0016, 0., -1., 0.], [0., 0., 0., 1.]])
    htm7 = np.matrix([[1., 0., 0., 0.], [0., 0.0008, -1., 0.], [0., 1., 0.0008, -0.012], [0., 0., 0., 1.]])
    
    base_3d_obj = [
        Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Base.obj',
                0.001,
                htm1,
                MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Base.mtl',
                             opacity=opacity))]

    link_3d_obj = []

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis1.obj',
                 0.001,
                 htm2,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis1.mtl',
                             opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis2.obj',
                 0.001,
                 htm3,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis2.mtl',
                             opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis3.obj',
                 0.001,
                 htm4,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis3.mtl',
                             opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis4.obj',
                 0.001,
                 htm5,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis4.mtl',
                             opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis5.obj',
                 0.001,
                 htm6,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis5.mtl',
                             opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis6.obj',
                 0.001,
                 htm7,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaKR5_updated/Axis6.mtl',
                             opacity=opacity))]
    )

    # Create links

    links = []
    for i in range(n):
        links.append(Link(i, link_info[0][i], link_info[1][i], link_info[2][i], link_info[3][i], link_info[4][i],
                          link_3d_obj[i]))

        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    q0 = [1.570, -1.570, 0.000, 0.000, 0, 0.000]

    #Create joint limits
    joint_limits = (np.pi/180)*np.matrix([[-170,170],[-190, 45],[-119-90, 165-90],[-190,190],[-120,120],[-358,358]])

    return base_3d_obj, links, np.identity(4), np.identity(4), q0, joint_limits
