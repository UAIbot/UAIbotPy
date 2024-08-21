from robot.links import Link
import robot as rb
from utils import Utils
from graphics.meshmaterial import MeshMaterial
from graphics.model3d import Model3D
from simobjects.rigidobject import RigidObject
from simobjects.pointlight import PointLight
from simulation import Simulation
from simobjects.frame import Frame
import numpy as np

def map_color(color):
    if isinstance(color, list):
        colors = color
        if len(color) < 3:
            print("Jaco has 3 colors. Repeating the last one provided.")
            while len(colors) < 3:
                colors.append(color[-1])
    
    elif color is None:
        colors = ["#3e3f42", "#919090", "#1d1d1f"]
    
    else:
        colors = [color]*3

    return colors

def _create_jaco(htm=np.identity(4), name='jaco', color=None, opacity=1):
    colors = map_color(color)
    color1, color2, color3 = colors

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception(
            "The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not (Utils.is_a_name(name)):
        raise Exception(
            "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")
    for color in colors:
        if (not Utils.is_a_color(color)):
            raise Exception("The parameter 'color' should be a HTML-compatible color.")

    if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
        raise Exception("The parameter 'opacity' should be a float between 0 and 1.")

    pi = np.pi
    d1 = 0.2755
    d2 = 0.41
    d3 = 0.2073
    d4 = 0.0741
    d5 = 0.0741
    d6 = 0.16
    e2 = 0.0098
    aa = 30*pi/180
    sa = np.sin(aa)
    s2a = np.sin(2*aa)
    d4b = d3 + (sa/s2a) * d4
    d5b = (sa/s2a)*d4 + (sa/s2a)*d5
    d6b = (sa/s2a)*d5 + d6

    jaco_DH_theta = np.array([0,     0,    0,    0,    0,    0])
    jaco_DH_d = np.array([d1,    0,  -e2, -d4b, -d5b, -d6b])
    jaco_DH_a = np.array([0,    d2,    0,    0,    0,    0])
    jaco_DH_alpha = np.array([pi/2, pi, pi/2, 2*aa, 2*aa,    pi])

    jaco_DH_type = np.array([0, 0, 0, 0, 0, 0])
    link_info = np.array(
        [jaco_DH_theta, jaco_DH_d, jaco_DH_alpha, jaco_DH_a, jaco_DH_type])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    mesh = MeshMaterial(metalness=0.8, clearcoat=0.9, roughness=0.3,
                        normal_scale=[0.5, 0.5], color=color1,
                        opacity=opacity, side="DoubleSide")
    mesh_ring = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45,
                             normal_scale=[0.5, 0.5], color=color2,
                             opacity=opacity, side="DoubleSide")
    mesh_nail = MeshMaterial(metalness=0, clearcoat=0, roughness=1,
                             normal_scale=[0.5, 0.5], color=color3,
                             opacity=opacity, side="DoubleSide")
   
    q0 = [pi, pi, pi, pi, 0, pi/2]

    Q00 = Utils.trn([0, 0, 0])
    Q01 = (Utils.rotz(link_info[0, 0] - pi/2 - q0[0]) * Utils.trn([0, 0, link_info[1, 0]])
           * Utils.rotx(link_info[2, 0]) * Utils.trn([link_info[3, 0], 0, 0]))
    Q02 = Q01 @ (Utils.rotz(link_info[0, 1] - pi/2 - np.deg2rad(25.002) + q0[1]) * Utils.trn(
        [0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn([link_info[3, 1], 0, 0]))
    Q03 = Q02 @ (Utils.rotz(link_info[0, 2] - pi/2 + np.deg2rad(25.002) + q0[2]) * Utils.trn(
        [0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn([link_info[3, 2], 0, 0]))
    Q04 = Q03 @ (Utils.rotz(link_info[0, 3] + pi/2 + q0[3]) * Utils.trn(
        [0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn([link_info[3, 3], 0, 0]))
    Q05 = Q04 @ (Utils.rotz(link_info[0, 4] + pi + q0[4]) * Utils.trn(
        [0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn([link_info[3, 4], 0, 0]))
    Q06 = Q05 @ Utils.rotz(link_info[0, 5] + pi/2 + q0[5]) * Utils.trn(
        [0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn([link_info[3, 5], 0, 0])

    link0_mth = Utils.inv_htm(Q00)
    base_3d_obj = [
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/base.obj',
                scale=scale, htm=link0_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/base_ring.obj',
                scale=scale, htm=link0_mth, mesh_material=mesh_ring)]
    
    link_3d_obj = []
    
    # Shoulder
    link1_mth = Utils.inv_htm(Q01)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/shoulder.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/shoulder_ring.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh_ring),
    ])

    # Upper arm + elbow
    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/upperarm.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/upperarm_ring.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh_ring),
    ])

    # Forearm
    link3_mth = Utils.inv_htm(Q03)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/forearm.obj',
                scale=scale, htm=link3_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/forearm_ring.obj',
                scale=scale, htm=link3_mth, mesh_material=mesh_ring),
    ])

    link4_mth = Utils.inv_htm(Q04)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist1.obj',
                scale=scale, htm=link4_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist1_ring.obj',
                scale=scale, htm=link4_mth, mesh_material=mesh_ring),
    ])

    link5_mth = Utils.inv_htm(Q05)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist2.obj',
                scale=scale, htm=link5_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/wrist2_ring.obj',
                scale=scale, htm=link5_mth, mesh_material=mesh_ring),
    ])

    link6_mth = Utils.inv_htm(Q06)
    link_3d_obj.append([
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/gripper.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/handpalm.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_mounting.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_proximal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_distal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger1_nail.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_nail),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_mounting.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_proximal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_distal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/finger2_nail.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_nail),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_mounting.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_proximal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_distal.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_ring),
        Model3D(url='https://raw.githubusercontent.com/fbartelt/robotics-experiments/main/models/jaco/thumb_nail.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh_nail),
    ])

    # com_coordinates = [[0.534615, 0, 0.15], [1.5353, 0, 0.15]]
    com_coordinates = [np.eye(4)[:-1, :] @ (Utils.inv_htm(Q01) @ np.array([-3.11506292e-3,  1.62075358e-5,  2.66810879e-1, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q02) @ np.array(
                           [-0.00592762,  0.14709695,  0.5909634, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q03) @ np.array(
                           [0.01162087, 0.04102689, 0.53732661, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q04) @ np.array(
                           [0.00971901, -0.01057426,  0.44918022, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q05) @ np.array(
                           [0.0097224, -0.03312561,  0.3785306, 1]).reshape(-1, 1)),
                       np.eye(4)[:-1, :] @ (Utils.inv_htm(Q06) @ np.array([0.0033009, -0.09643814,  0.32164165, 1]).reshape(-1, 1))]
    list_inertia_mat = []

    # Use parameters in MENDES, M. P. Computed torque-control of the Kinova JACO2 Arm. 2017
    list_mass = [0.767664, 0.99571300239, 0.79668391394,
                0.41737, 0.41737, 1.07541]
    I1 = np.array([[0.002231083822876,  0.000006815637176,  -0.000019787600863], 
                    [0,  0.000620733840723, -0.000306288710418], 
                    [0, 0,  0.002398007441187]])
    I1 = np.triu(I1, 1).T + I1
    list_inertia_mat.append(I1)
    I2 = np.array([[0.004162524943115,  -0.000000552736891,  -0.001487468585390], 
                    [0,  0.025495429281017, -0.000000277011102], 
                    [0, 0,  0.021736935450151]])
    I2 = np.triu(I2, 1).T + I2
    list_inertia_mat.append(I2)
    I3 = np.array([[0.002861254222538,  0.000000402773293,  0.000000566406981], 
                    [0,  0.002738656386387, -0.000344659168888], 
                    [0, 0,  0.000351242752544]])
    I3 = np.triu(I3, 1).T + I3
    list_inertia_mat.append(I3)
    I4 = np.array([[0.708476728234e-3,  0.008271643300e-3,  0.112833961462e-3], 
                    [0,  0.740496291632e-3, 0.000494273498e-3], 
                    [0, 0,  0.178193295188e-3]])
    I4 = np.triu(I4, 1).T + I4
    list_inertia_mat.append(I4)
    I5 = np.array([[0.827477604255e-3,  0.008281078147e-3,  -0.101690165377e-3], 
                    [0,  0.852081770282e-3, -0.000054222717e-3], 
                    [0, 0,  0.170777897817e-3]])
    I5 = np.triu(I5, 1).T + I5
    list_inertia_mat.append(I5)
    I6 = np.array([[0.004833755945304,  0.000004331179432,  0.000361596139440], 
                    [0,  0.004841503493061, -0.000002455131406], 
                    [0, 0,  0.000199821519482]])
    I6 = np.triu(I6, 1).T + I6
    list_inertia_mat.append(I6)
        
    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i], com_coordinates=com_coordinates[i], mass=list_mass[i], inertia_matrix=list_inertia_mat[i]))

    htm_n_eef = Utils.rotz(-pi) * Utils.rotx(0.3056*pi) * \
        Utils.rotx(0.3056*pi) * Utils.trn([0, 0, 0.052])
    htm_n_eef = Utils.trn([0, 0, 0])
    htm_base_0 = Utils.trn([0, 0, 0])

    # Create joint limits
    joint_limits = np.matrix([[-3*np.pi, 3*np.pi], [-np.deg2rad(47), np.deg2rad(266)],
                              [-np.deg2rad(19), np.deg2rad(322)
                               ], [-3*np.pi, 3*np.pi],
                              [-3*np.pi, 3*np.pi], [-3*np.pi, 3*np.pi]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits