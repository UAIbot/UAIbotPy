import numpy as np
from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.mtlmeshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.ball import *
from uaibot.simobjects.box import *
from uaibot.simobjects.cylinder import *

from .links import *


def _create_franka_emika_3(htm, name, color, opacity):
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color) and not color=='':
        raise Exception("The parameter 'color' should be a HTML-compatible color or empty.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")



    link_info = [[0.00000, 0.0000, 0.0000, 0.00000, 0.00000, 0.0000, 0.0000],
                 [0.33300, 0.0000, 0.3160, 0.00000, 0.38400, 0.0000, 0.1070+0.07],  # "d" translation in z
                 [-1.5708, 1.5708, 1.5708, -1.5708, 1.57080, 1.5708, 0.0000],  # "alfa" rotation in x
                 [0.00000, 0.0000, 0.0825, -0.0825, 0.00000, 0.0880, 0.0000],  # "a" translation in x 0.25
                 [0, 0, 0, 0, 0, 0, 0]]

    n = 7
    scale = 1


    # Collision model
    col_model = [[], [], [], [], [], [], []]

    col_model[0].append(Cylinder(htm= Utils.trn([0,0,0]),
                                 name=name + "_C0_0", radius=0.08, height=0.28, color="red", opacity=0.3))

    col_model[1].append(Cylinder(htm= Utils.trn([0,0,0.18]),
                                 name=name + "_C1_0", radius=0.07, height=0.25, color="blue", opacity=0.3))
        
    col_model[2].append(Cylinder(htm= Utils.trn([0,0,0]),
                                 name=name + "_C2_0", radius=0.07, height=0.24, color="green", opacity=0.3))

    col_model[3].append(Cylinder(htm= Utils.trn([0,0,0.13]),
                                 name=name + "_C3_0", radius=0.07, height=0.20, color="yellow", opacity=0.3))

    A = np.matrix([ [1,0,0,0], [0,0,1,-0.383], [0,-1,0,0],[0,0,0,1]])

    col_model[4].append(Box(htm= A * Utils.trn([0,0.09,0.28]) * Utils.rotx(-0.24),
                                 name=name + "_C4_0", width=0.08, depth=0.05, height=0.23, color="magenta", opacity=0.3))

    col_model[4].append(Cylinder(htm= Utils.trn([0,0,-0.03]),
                                 name=name + "_C4_1", radius=0.06, height=0.22, color="magenta", opacity=0.3))

    col_model[5].append(Cylinder(htm= Utils.trn([0,0,0.02]),
                                 name=name + "_C5_0", radius=0.06, height=0.21, color="cyan", opacity=0.3))

    col_model[6].append(Box(htm= Utils.trn([0,0,0.04-0.09]), 
                            name=name + "_C6_0", width=0.05 , depth=0.21, height=0.10, color="red", opacity=0.3))

    col_model[6].append(Box(htm= Utils.trn([0.03, 0.03, -0.025-0.07]) * Utils.rotz(np.pi / 4), 
                            name=name + "_C6_1", width=0.1000 , depth=0.09, height=0.0340, color="red", opacity=0.3))


    # Create 3d objects
    htm1 = np.matrix([[1., 0., 0., 0.], [0., 0.0008, -1., 0.], [0., 1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm2 = np.matrix([[1., 0., 0., 0.], [0., 0.0008, 1., 0.], [0., -1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm3 = np.matrix([[1., 0., 0., -0.0825], [0., 0.0008, 1., 0.], [0., -1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm4 = np.matrix([[1., 0., 0., 0.0825], [0., 0.0008, -1., 0.], [0., 1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm5 = np.matrix([[1., 0., 0., 0.], [0., 0.0008, 1., 0.], [0., -1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm6 = np.matrix([[1., 0., 0., -0.088], [0., 0.0008, 1., 0.], [0., -1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm7 = np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -0.177], [0., 0., 0., 1.]])
    htm8 = np.matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., -0.067], [0., 0., 0., 1.]])  

    link_3d_obj = []
    
    default_material = MeshMaterial(metalness=0.3, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5],color=color, opacity=opacity)

    base_3d_obj = [Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link0.obj',
        scale,
        np.identity(4),
        MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link0.mtl', 
                        opacity=opacity) if color=='' else default_material) ]
    
    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link1.obj',
            scale,
            htm1,
            MTLMeshMaterial(url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link1.mtl',
                         opacity=opacity) if color=='' else default_material)
        ]
    )
        

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link2.obj',
            scale,
            htm2,
            MTLMeshMaterial(url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link2.mtl',
                         opacity=opacity) if color=='' else default_material)
        ]
    )
    

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link3.obj',
            scale,
            htm3, 
            MTLMeshMaterial(url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link3.mtl',
                         opacity=opacity) if color=='' else default_material)
        ]
    )

 
    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link4.obj',
            scale,
            htm4, 
            MTLMeshMaterial(url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link4.mtl',
                         opacity=opacity) if color=='' else default_material)
        ]
    )
            
    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link5.obj',
            scale,
            htm5, 
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link5.mtl',
                         opacity=opacity) if color=='' else default_material)
        ]
    )
 

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link6.obj',
            scale,
            htm6, 
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link6.mtl',
                         opacity=opacity) if color=='' else default_material)
        ]
    )


    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link7.obj',
            scale,
            htm7, 
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/link7.mtl',
                         opacity=opacity) if color=='' else default_material),
        Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/hand.obj',
            scale,
            htm8,  
            MTLMeshMaterial(url = 'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/FrankaEmikaPanda/hand.mtl',
                            opacity=opacity) if color=='' else default_material)
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
    q0 = [0.0, 0.0, 0.0, -np.pi*4/180, 0.0, 0.0, 0.0]

    #Create joint limits
    joint_limits = (np.pi/180)*np.matrix([[-166,166],[-101,101],[-166,166],[-176,-4],[-166,166],[-1,215],[-166,166]])
    
    return base_3d_obj, links, np.identity(4), np.identity(4), q0, joint_limits