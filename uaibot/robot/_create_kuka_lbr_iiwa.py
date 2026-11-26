from uaibot.utils import *

from uaibot.graphics.meshmaterial import *
from uaibot.graphics.glbmeshmaterial import *
from uaibot.graphics.model3d import *

from uaibot.simobjects.ball import *
from uaibot.simobjects.box import *
from uaibot.simobjects.cylinder import *

from .links import *


def _create_kuka_lbr_iiwa(htm, name, color, opacity):
    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

    if not Utils.is_a_color(color):
        raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    link_info = [[0, 0, 0, 0, 0, 0, 0],
                 [0.36, 0, 0.42, 0, 0.4, 0, 0.152],  # "d" translation in z
                 [-3.14 / 2, 3.14 / 2, 3.14 / 2, -3.14 / 2, -3.14 / 2, 3.14 / 2, 0],  # "alfa" rotation in x
                 [0, 0, 0, 0, 0, 0, 0],  # "a" translation in x 0.25
                 [0, 0, 0, 0, 0, 0, 0]]

    ############
    n = 7
    scale = 1
    dy=-0.02

    # Collision model
    col_model = [[], [], [], [], [], [], []]

    col_model[0].append(Cylinder(htm=np.matrix([[1,0,0,0] , [0,0,-1,0.19-dy], [0,1,0,0], [0,0,0,1]]) ,
                                 name=name + "_C0_0", radius=0.10, height=0.21, color="red", opacity=0.3))
    col_model[0].append(Ball(htm=Utils.trn([0.01, 0.01-dy, 0]),
                             name=name + "_C0_1", radius=0.13, color="red", opacity=0.3))

    col_model[1].append(Ball(htm=Utils.trn([0, 0.03, 0.04+dy]),
                             name=name + "_C1_0", radius=0.12, color="green", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([0, 0.02, 0.2+dy+0.0125]),
                                 name=name + "_C1_1", radius=0.09, height=0.145, color="green", opacity=0.3))

    col_model[2].append(Cylinder(htm=np.matrix([[1,0,0,0] , [0,0,1,-0.08+dy], [0,-1,0,0], [0,0,0,1]]),
                                 name=name + "_C2_0", radius=0.08, height=0.1, color="blue", opacity=0.3))
    col_model[2].append(Ball(htm=Utils.trn([0, -0.01+dy, -0.01]),
                             name=name + "_C2_1", radius=0.115, color="blue", opacity=0.3))

    col_model[3].append(Ball(htm=Utils.trn([0, -0.035, 0.063+2*dy]),
                             name=name + "_C3_0", radius=0.09, color="orange", opacity=0.3))
    col_model[3].append(Cylinder(htm=Utils.trn([0, -0.01, 0.195+2*dy]),
                                 name=name + "_C3_1", radius=0.09, height=0.125, color="orange", opacity=0.3))

    col_model[4].append(Cylinder(htm=np.matrix([[1,0,0,0] , [0,0,-1,0.088-2*dy], [0,1,0,0], [0,0,0,1]]),
                                 name=name + "_C4_0", radius=0.075, height=0.11, color="magenta", opacity=0.3))
    col_model[4].append(Box(htm=np.matrix([[1,0,0,0] , [0,0,-1,-0.01-2*dy], [0,1,0,-0.08], [0,0,0,1]]),
                            name=name + "_C4_1", width=0.12, depth=0.04, height=0.20, color="magenta", opacity=0.3))

    col_model[5].append(Cylinder(htm=Utils.trn([0, 0, 0.063+2*dy]),
                                 name=name + "_C5_0", radius=0.075, height=0.18, color="cyan", opacity=0.3))


    # Create 3d objects
    htm1 = np.matrix([[1., 0., 0., 0.], [0., 0.0008, -1., 0.], [0., 1., 0.0008, 0.], [0., 0., 0., 1.]])
    htm2 = np.matrix([[1., 0., -0.0016, 0.], [0., -1., 0., 0.36], [-0.0016, 0., -1., 0.], [0., 0., 0., 1.]])
    htm3 = np.matrix([[1., 0., -0.0016, 0.], [-0.0016, 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    htm4 = np.matrix([[1., 0., -0.0016, 0.], [0., 1., 0., -0.42], [0.0016, 0., 1., 0.], [0., 0., 0., 1.]])
    htm5 = np.matrix([[1., 0., -0.0016, 0.], [-0.0016, 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    htm6 = np.matrix([[1., 0., 0., 0.], [0., -1., 0., 0.4], [0., 0., -1., 0.], [0., 0., 0., 1.]])
    htm7 = np.matrix([[1., 0., 0., 0.], [0., 0., -1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
    htm8 = np.matrix([[1., 0., 0., 0.], [0., 0.0008, -1., 0.], [0., 1., 0.0008, -0.17], [0., 0., 0., 1.]])
    
    default_material_0 = MeshMaterial(metalness=0.3, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5],
                     color=color, opacity=opacity)
    default_material_1 = MeshMaterial(metalness=0.7, clearcoat=1, roughness=0.5, normal_scale=[0.5, 0.5], color="#707070",
                         opacity=opacity)
    
    base_3d_obj = [Model3D(
        'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_base_link.glb',
        scale,
        htm1 ,
        GLBMeshMaterial(opacity=opacity) if color == '' else default_material_0)]

    link_3d_obj = []

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_link_1.glb',
            scale,
            htm2,
            GLBMeshMaterial(opacity=opacity) if color == '' else default_material_0)
        ]
    )


    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_link_2.glb',
            scale,
            htm3,
            GLBMeshMaterial(opacity=opacity) if color == '' else default_material_0)
        ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_link_3.glb',
            scale,
            htm4,
            GLBMeshMaterial(opacity=opacity) if color == '' else default_material_0)
        ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_link_4.glb',
            scale,
            htm5,
            GLBMeshMaterial(opacity=opacity) if color == '' else default_material_0)
        ]
    )

    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_link_5.glb',
            scale,
            htm6 ,
            GLBMeshMaterial(opacity=opacity) if color == '' else default_material_0)
        ]
    )
    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_link_6.glb',
            scale,
            htm7,
            GLBMeshMaterial(opacity=opacity) if color == '' else default_material_0)
        ]
    )
    link_3d_obj.append(
        [Model3D(
            'https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/RobotModels/KukaLBRIIWA/iiwa_link_7.glb',
            scale,
            htm8,
            GLBMeshMaterial(opacity=opacity) if color == '' else default_material_1)
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

    q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    #Create joint limits
    joint_limits = (np.pi/180)*np.matrix([[-170,170],[-120,120],[-170,170],[-120,120],[-170,170],[-120,120],[-175,175]])
  
    return base_3d_obj, links, np.identity(4), Utils.trn([0,0,-0.04]), q0, joint_limits
