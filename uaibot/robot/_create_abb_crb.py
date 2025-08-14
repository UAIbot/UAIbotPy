from utils import *

from graphics.meshmaterial import *
from graphics.model3d import *
from graphics.mtlmeshmaterial import *

from simobjects.ball import *
from simobjects.box import *
from simobjects.cylinder import *

from .links import *


def _create_abb_crb(htm, name, color, opacity):
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    link_info = [[       0,     0,      3.14,         0, 0, 0],  # "theta" rotation in z
                 [   0.265,     0,         0,     -0.47, 0, 0.101],  # "d" translation in z
                 [3.14 / 2,     0, -3.14 / 2, -3.14 / 2, -3.14 / 2, 0],  # "alfa" rotation in x
                 [       0, 0.444,      0.11,         0, 0.08, 0],  # "a" translation in x
                 [       0,      0,        0,         0, 0, 0]]  # joint type

    scale = 1
    n = 6

    # Collision model
    col_model = [[], [], [], [], [], []]

    col_model[0].append(Cylinder(htm= Utils.trn([0,0,0.025]),
                                 name=name + "_C0_0", radius=0.08, height=0.30, color="red", opacity=0.3))

    col_model[1].append(Cylinder(htm= Utils.trn([0,0,0.025]),
                                 name=name + "_C1_0", radius=0.08, height=0.30, color="blue", opacity=0.3))

    col_model[1].append(Box(htm= Utils.trn([-0.23,0,0.025+0.11]),
                                 name=name + "_C1_1", width=0.58, depth = 0.15, height=0.08, color="blue", opacity=0.3))

    col_model[2].append(Ball(htm= Utils.trn([-0.01,0,-0.065]),
                                 name=name + "_C2_0", radius=0.075, color="green", opacity=0.3))

    col_model[3].append(Cylinder(htm= Utils.trn([0,-0.3,0])*Utils.rotx(np.pi/2),
                                 name=name + "_C3_0", radius=0.065, height=0.20, color="yellow", opacity=0.3))

    col_model[3].append(Box(htm= Utils.trn([0,-0.1,-0.09]),
                                 name=name + "_C3_1", width=0.11, depth=0.32, height=0.08, color="yellow", opacity=0.3))

    col_model[3].append(Cylinder(htm= Utils.trn([0,0,-0.01]),
                                 name=name + "_C3_2", radius=0.065, height=0.20, color="yellow", opacity=0.3))

    col_model[4].append(Cylinder(htm= Utils.trn([0,0,-0.01]),
                                 name=name + "_C4_0", radius=0.065, height=0.24, color="magenta", opacity=0.3))
                          
    # Create 3d objects

    htm0 = np.matrix([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    htm1 = np.matrix([[1., 0., 0., 0.], [0., 1., -0.0008, 0.], [0., 0.0008, 1., 0.], [0., 0., 0., 1.]])
    htm2 = np.matrix([[0.0008, 1., 0., -0.444], [-1., 0.0008, -0.0008, 0.], [-0.0008, -0., 1., 0.], [0., 0., 0., 1.]])
    htm3 = np.matrix([[0.0008, 1., 0., -0.11], [0., 0., -1., 0.], [-1., 0.0008, 0., 0.], [0., 0., 0., 1.]])
    htm4 = np.matrix([[0., -1., 0.0008, 0.], [1., 0., -0.0008, -0.47], [0.0008, 0.0008, 1., 0.], [0., 0., 0., 1.]])
    htm5 = np.matrix([[0., 1., 0.0016, -0.08], [0.0008, -0.0016, 1., 0.], [1., 0., -0.0008, 0.], [0., 0., 0., 1.]])
    htm6 = np.matrix([[0.0008, -1., -0., 0.], [0., 0., -1., 0.], [1., 0.0008, 0., 0.], [0., 0., 0., 1.]])



    base_3d_obj = [
        Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/base_link.obj',
            scale,
            htm0,
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/base_link.mtl', 
                        opacity=opacity))]

    link_3d_obj = []

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link1.obj',
                 scale,
                 htm1,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link1.mtl', 
                        opacity=opacity))]
   
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link2.obj',
                 scale,
                 htm2,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link2.mtl', 
                        opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link3.obj',
                 scale,
                 htm3,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link3.mtl', 
                        opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link4.obj',
                 scale,
                 htm4,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link4.mtl', 
                        opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link5.obj',
                 scale,
                 htm5,
                 MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link5.mtl', 
                        opacity=opacity))]
    )

    link_3d_obj.append(
        [Model3D('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/link6.obj',
                 scale,
                 htm6,
                MeshMaterial(color="#BEBEBE",
                             metalness=1.0, 
                             clearcoat=1, 
                             clearcoat_roughness=0.2, 
                             roughness=0.5, 
                             normal_scale=[0.5, 0.5], 
                             normal_map="https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/ABBCRB15000/metallic_lines.png",
                             opacity=opacity,
                        emissive="#555454", 
                        emissive_intensity=0.1,
                    )
                )]
    )

    # Create links

    links = []
    for i in range(n):
        links.append(Link(i, link_info[0][i], link_info[1][i], link_info[2][i], link_info[3][i], link_info[4][i],
                          link_3d_obj[i]))

        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    q0 = [0.000, 0.000, 3.14, 0.000, 0.000, 0.000]

    #Create joint limits
    joint_limits = (np.pi/180)*np.matrix([[-180,180],[-180,180],[-225-200+10,85-200-20],[-180,180],[-180,180],[-180,180]])


    return base_3d_obj, links, np.identity(4), np.identity(4), q0, joint_limits