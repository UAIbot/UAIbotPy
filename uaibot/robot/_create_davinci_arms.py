from utils import Utils

from graphics.meshmaterial import MeshMaterial
from graphics.model3d import Model3D
from simobjects.box import Box
from simobjects.cylinder import Cylinder
from robot.links import Link

import numpy as np

def _create_davinci_arm1(color=["#919090", "#61b3c7", "#a6a6a6"], opacity=1, name='davinci_arm1'):
    color1, color2, color3 = color

    theta2 = np.deg2rad(1.875)
    theta3 = np.deg2rad(3.25 - 1.875)
    theta4 = np.deg2rad(75.25)
    theta5 = np.deg2rad(28.75 + 90)
    theta6 = np.deg2rad(-45)
    theta7 = np.deg2rad(94 + 180)
    theta8 = np.deg2rad(-170.5)
    theta9 = np.deg2rad(61.5)

    d3 = -96e-3
    d5 = -96e-3 * 1.9
    d6 = (431.8 * 1.417)/1000
    d9 = -3.75e-2
    d10 = 3e-2

    alpha5 = np.pi*(1/2 + 1/9)
    alpha6 = np.pi/2
    alpha9 = np.pi/2

    a2 = 0.3
    a3 = 0.415
    a4 = 0.415
    a6 = 3.1e-3
    a7 = 0.27
    a8 = 0.474
    a9 = 0.098

    # Passive joints 1, 2, 3, 4, 5, 8, 9
    # Active joints 6, 7, 10
    link_info = np.array([
        # "theta" rotation in z
        [0, 0,   0,  0,      0,      0,  0,  0,      0,   0],
        # "d" translation in z
        [0, 0,  d3,  0,     d5,     d6,  0,  0,     d9, d10],
        # "alfa" rotation in x
        [0, 0,   0,  0, alpha5, alpha6,  0,  0, alpha9,   0],
        # "a" translation in x
        [0, a2, a3, a4,      0,     a6, a7, a8,     a9,   0],
        # joint type
        [1, 0,   0,  0,      0,      0,  0,  0,      0,   1]
    ])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    link_3d_obj = []
    mesh = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color1, opacity=opacity, side="DoubleSide")
    mesh2 = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color2, opacity=opacity, side="DoubleSide")
    mesh3 = MeshMaterial(metalness=0.8, roughness=0.4, clearcoat=0, normal_scale=[
                        0.5, 0.5], color=color3, opacity=opacity, side="DoubleSide")
    b1 = 0.1
    b2 = 0.45
    b3 = 1.1

    Q00 = Utils.trn([b1, b3, -b2]) * Utils.rotx(-np.pi/2) * \
        Utils.rotz(np.pi)  # change reference frame
    Q01 = Q00 * Utils.rotz(link_info[0, 0]) * Utils.trn([0, 0, link_info[1, 0]]) * Utils.rotx(link_info[2, 0]) * Utils.trn(
        [link_info[3, 0], 0, 0])
    Q02 = Q01 * (Utils.rotz(link_info[0, 0] + theta2) * Utils.trn([0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn(
        [link_info[3, 1], 0, 0]))
    Q03 = Q02 * (Utils.rotz(link_info[0, 1] + theta3) * Utils.trn([0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn(
        [link_info[3, 2], 0, 0]))
    Q04 = Q03 * (Utils.rotz(link_info[0, 2] + theta4) * Utils.trn([0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn(
        [link_info[3, 3], 0, 0]))
    Q05 = Q04 * (Utils.rotz(link_info[0, 3] + theta5) * Utils.trn([0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn(
        [link_info[3, 4], 0, 0]))
    Q06 = Q05 * (Utils.rotz(link_info[0, 4] + theta6) * Utils.trn([0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn(
        [link_info[3, 5], 0, 0]))
    Q07 = Q06 * (Utils.rotz(link_info[0, 5] + theta7) * Utils.trn([0, 0, link_info[1, 6]]) * Utils.rotx(link_info[2, 6]) * Utils.trn(
        [link_info[3, 6], 0, 0]))
    Q08 = Q07 * (Utils.rotz(link_info[0, 6] + theta8) * Utils.trn([0, 0, link_info[1, 7]]) * Utils.rotx(link_info[2, 7]) * Utils.trn(
        [link_info[3, 7], 0, 0]))
    Q09 = Q08 * (Utils.rotz(link_info[0, 7] + theta9) * Utils.trn([0, 0, link_info[1, 8]]) * Utils.rotx(link_info[2, 8]) * Utils.trn(
        [link_info[3, 8], 0, 0]))
    Q010 = Q09 * (Utils.rotz(link_info[0, 8]) * Utils.trn([0, 0, link_info[1, 9]]) * Utils.rotx(link_info[2, 9]) * Utils.trn(
        [link_info[3, 9], 0, 0]))

    link1_mth = Utils.inv_htm(Q01)
    link_3d_obj.append([
        # torre movel
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/4.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
        # cilindro conector
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/5.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
    ])

    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        # ligacao de cilindros curta
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/8.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
    ])

    link3_mth = Utils.inv_htm(Q03)
    link_3d_obj.append([
        # cilindro conector
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/59.obj',
            scale=scale, htm=link3_mth, mesh_material=mesh),
        # ligacao de cilindros longa
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/9.obj',
            scale=scale, htm=link3_mth, mesh_material=mesh),
    ])

    link4_mth = Utils.inv_htm(Q04)
    link_3d_obj.append([
        # ligacao de cilindros longa
        Model3D(

            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/11.obj',
            scale=scale, htm=link4_mth, mesh_material=mesh),
    ])

    link5_mth = Utils.inv_htm(Q05)
    link_3d_obj.append([
        # cotovelo
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/10.obj',
            scale=scale, htm=link5_mth, mesh_material=mesh),
    ])

    link6_mth = Utils.inv_htm(Q06)
    link_3d_obj.append([
        # pá dobrada
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/38.obj',
            scale=scale, htm=link6_mth, mesh_material=mesh),
    ])

    link7_mth = Utils.inv_htm(Q07)
    link_3d_obj.append([
        # conecta pá com bumerangue
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/19.obj',
            scale=scale, htm=link7_mth, mesh_material=mesh),
    ])

    link8_mth = Utils.inv_htm(Q08)
    link_3d_obj.append([
        # bumerangue
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/21.obj',
            scale=scale, htm=link8_mth, mesh_material=mesh),

    ])

    link9_mth = Utils.inv_htm(Q09)
    link_3d_obj.append([
        # envoltoria agulha
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/43.obj',
            scale=scale, htm=link9_mth, mesh_material=mesh),
    ])

    link10_mth = Utils.inv_htm(Q010)
    link_3d_obj.append([
        # guia
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/45.obj',
                scale=scale, htm=link10_mth, mesh_material=mesh),
        # agulha
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/54.obj',
                scale=scale, htm=link10_mth, mesh_material=mesh3),
        # abridor de garrafa
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/51.obj',
            scale=scale, htm=link10_mth, mesh_material=mesh),
        # botao
        Model3D(
            url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/52.obj',
                scale=scale, htm=link10_mth, mesh_material=mesh2),
    ])

    col_model = [[], [], [], [], [], [], [], [], [], []]

    col_model[0].append(Box(htm=Utils.trn([0, 0.02, -0.33]),
                            name=name + "_C0_0", width=0.14, height=0.4, depth=0.2, color="red", opacity=0.3))
    col_model[0].append(Cylinder(htm=Utils.trn([0, 0, -0.14]) @ Utils.rotz(3.14 / 2) @ Utils.rotx(3.14),
                                 name=name + "_C0_1", radius=0.075, height=0.19, color="red", opacity=0.3))

    col_model[1].append(Cylinder(htm=Utils.trn([0, 0, 0.02]) @ Utils.rotz(np.pi/2) @ Utils.rotx(np.pi),
                                 name=name + "_C1_0", radius=0.075, height=0.12, color="black", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([-0.3, 0, 0.02]) @ Utils.rotz(np.pi/2) @ Utils.rotx(np.pi),
                                 name=name + "_C1_1", radius=0.075, height=0.12, color="black", opacity=0.3))
    col_model[1].append(Box(htm=Utils.trn([-0.15, 0, 0.02]) @ Utils.roty(np.pi/2),
                            name=name + "_C1_2", width=0.11, height=0.3, depth=0.12, color="black", opacity=0.3))

    col_model[2].append(Box(htm=Utils.trn([-0.200, 0, -0.12]) @ Utils.roty(np.pi/2),
                            name=name + "_C2_0", width=0.11, height=0.4, depth=0.12, color="green", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([0, 0, -0.12]) @ Utils.rotz(np.pi/2) @ Utils.rotx(np.pi),
                                 name=name + "_C2_1", radius=0.075, height=0.12, color="green", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([-0.415, 0, -0.065]) @ Utils.rotz(np.pi/2) @ Utils.rotx(np.pi),
                                 name=name + "_C2_2", radius=0.075, height=0.23, color="green", opacity=0.3))

    col_model[3].append(Box(htm=Utils.trn([-0.200, 0, 0]) @ Utils.roty(np.pi/2),
                            name=name + "_C3_0", width=0.1, height=0.4, depth=0.12, color="blue", opacity=0.3))
    col_model[3].append(Cylinder(htm=Utils.trn([0, 0, -0.005]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C3_1", radius=0.075, height=0.15, color="blue", opacity=0.3))
    col_model[3].append(Cylinder(htm=Utils.trn([-0.415, 0, 0.01]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C3_2", radius=0.075, height=0.13, color="blue", opacity=0.3))

    col_model[4].append(Cylinder(htm=Utils.trn([0, 0, 0]) @ Utils.rotx(-alpha5),
                                 name=name + "_C4_0", radius=0.075, height=0.2, color="magenta", opacity=0.3))
    col_model[4].append(Box(htm=Utils.trn([0, -0.02, 0.12]) @ Utils.roty(np.pi/2),
                            name=name + "_C4_1", width=0.2, height=0.14, depth=0.14, color="magenta", opacity=0.3))

    col_model[5].append(Box(htm=Utils.trn([0, -0.355, -0.05]) @ Utils.roty(np.pi/2),
                            name=name + "_C5_0", width=0.2, height=0.08, depth=0.05, color="orange", opacity=0.3))
    col_model[5].append(Box(htm=Utils.trn([0, -0.15, -0.125]) @ Utils.roty(np.pi/2),
                            name=name + "_C5_1", width=0.05, height=0.09, depth=0.4, color="orange", opacity=0.3))

    col_model[6].append(Box(htm=Utils.trn([-0.2, 0.025, -0.028]),
                            name=name + "_C6_0", width=0.25, height=0.14, depth=0.15, color="cyan", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.02, 0.025, -0.09]),
                            name=name + "_C6_1", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.019, 0.023, 0.018]),
                            name=name + "_C6_2", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))

    col_model[7].append(Box(htm=Utils.trn([-0.42, 0.08, -0.0365]) @ Utils.rotz(np.pi/3),
                            name=name + "_C7_0", width=0.23, height=0.082, depth=0.13, color="#88264a", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.22, 0.08, -0.0365]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C7_1", width=0.11, height=0.09, depth=0.39, color="#88264a", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.023, 0.01, -0.073]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C7_2", width=0.08, height=0.01, depth=0.04, color="#88264a", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.023, 0.01, 0.0]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C7_3", width=0.08, height=0.01, depth=0.04, color="#88264a", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([0, 0, 0.0]),
                                 name=name + "_C7_4", radius=0.035, height=0.01, color="#88264a", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([0, -0.001, -0.073]),
                                 name=name + "_C7_5", radius=0.035, height=0.01, color="#88264a", opacity=0.3))

    col_model[8].append(Box(htm=Utils.trn([-0.052, -0.001, -0.042]),
                            name=name + "_C8_0", width=0.02, height=0.31, depth=0.12, color="Brown", opacity=0.3))
    col_model[8].append(Cylinder(htm=Utils.trn([-0.1, 0, 0]) @ Utils.rotx(np.pi/2),
                                 name=name + "_C8_1", radius=0.035, height=0.056, color="Brown", opacity=0.3))
    col_model[8].append(Box(htm=Utils.trn([-0.08, 0, 0]),
                            name=name + "_C8_2", width=0.04, height=0.07, depth=0.056, color="Brown", opacity=0.3))
    col_model[8].append(Box(htm=Utils.trn([-0.03, 0, 0.075]),
                            name=name + "_C8_3", width=0.04, height=0.05, depth=0.04, color="Brown", opacity=0.3))
    col_model[8].append(Cylinder(htm=Utils.trn([0, 0, 0.075]),
                                 name=name + "_C8_4", radius=0.02, height=0.056, color="Brown", opacity=0.3))

    col_model[9].append(Box(htm=Utils.trn([-0.005, 0.0, -0.2]),
                            name=name + "_C9_0", width=0.07, height=0.4, depth=0.12, color="MidnightBlue", opacity=0.3))
    col_model[9].append(Cylinder(htm=Utils.trn([0, 0, 0.163]),
                                 name=name + "_C9_1", radius=0.01, height=0.4, color="MidnightBlue", opacity=0.3))

    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i]))
        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    #     1  2  3  4  5  6  7  8  9 10
    q0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    htm_n_eef = Utils.trn([0, 0, 0.3025])
    htm_base_0 = Utils.trn([-b1, -b2, b3])

    # Create joint limits
    joint_limits = np.matrix([[-0.11, 0.636], [-np.pi, np.pi], [-np.pi, np.pi], [-np.deg2rad(162), np.deg2rad(162)], [-np.pi, np.pi],
                             [-np.pi, np.pi], [-np.pi, np.pi], [-np.deg2rad(175), np.deg2rad(50)], [-np.deg2rad(108), np.deg2rad(70)], [-0.08, 0.05]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits


def _create_davinci_arm2(color=["#919090", "#61b3c7", "#a6a6a6"], opacity=1, name='davinci_arm2'):
    color1, color2, color3 = color

    theta2 = np.deg2rad(-3)
    theta3 = np.deg2rad(98.5)
    theta4 = np.deg2rad(108.5)
    theta5 = np.deg2rad(-44)
    theta6 = np.deg2rad(-85.5)
    theta7 = np.deg2rad(-171)
    theta8 = np.deg2rad(61.7)

    d2 = 96e-3 * 1.2
    d4 = -96e-3 * 0.925
    d5 = 431.8e-3 * 1.419
    d8 = -2.9e-2

    alpha4 = np.deg2rad(90 + 36.5)
    alpha5 = np.pi/2
    alpha8 = np.deg2rad(90)

    a2 = 0.415
    a3 = 0.415
    a5 = -8e-4
    a6 = 0.27
    a7 = 0.474
    a8 = 0.098

    # Passive joints 1, 2, 3, 4, 7, 8
    # Active joints 5, 6, 9
    link_info = np.array([
        # "theta" rotation in z
        [0,  0,   0,       0,      0,      0,      0,       0, 0],
        # 0,  "d" translation in z
        [0, d2,   0,      d4,     d5,      0,      0,      d8, 0],
        # 0,  "alfa" rotation in x
        [0,  0,   0,  alpha4, alpha5,      0,      0,  alpha8, 0],
        # 0,  "a" translation in x
        [0, a2,  a3,       0,     a5,     a6,     a7,      a8, 0],
        # 0,  joint type
        [1,  0,   0,       0,      0,      0,      0,       0, 1]
    ])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    link_3d_obj = []
    mesh = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color1, opacity=opacity, side="DoubleSide")
    mesh2 = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color2, opacity=opacity, side="DoubleSide")
    mesh3 = MeshMaterial(metalness=0.8, roughness=0.4, clearcoat=0, normal_scale=[
                        0.5, 0.5], color=color3, opacity=opacity, side="DoubleSide")
    b1 = -0.086
    b2 = 0.2634
    b3 = 1.3

    Q00 = Utils.trn([b1, b3, -b2]) * Utils.rotx(-np.pi/2) * \
        Utils.rotz(np.pi)  # change reference frame
    Q01 = Q00 * (Utils.rotz(link_info[0, 0]) * Utils.trn([0, 0, link_info[1, 0]]) * Utils.rotx(link_info[2, 0]) * Utils.trn(
        [link_info[3, 0], 0, 0]))
    Q02 = Q01 * (Utils.rotz(link_info[0, 0] + theta2) * Utils.trn([0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn(
        [link_info[3, 1], 0, 0]))
    Q03 = Q02 * (Utils.rotz(link_info[0, 1] + theta3) * Utils.trn([0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn(
        [link_info[3, 2], 0, 0]))
    Q04 = Q03 * (Utils.rotz(link_info[0, 2] + theta4) * Utils.trn([0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn(
        [link_info[3, 3], 0, 0]))
    Q05 = Q04 * (Utils.rotz(link_info[0, 3] + theta5) * Utils.trn([0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn(
        [link_info[3, 4], 0, 0]))
    Q06 = Q05 * (Utils.rotz(link_info[0, 4] + theta6) * Utils.trn([0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn(
        [link_info[3, 5], 0, 0]))
    Q07 = Q06 * (Utils.rotz(link_info[0, 5] + theta7) * Utils.trn([0, 0, link_info[1, 6]]) * Utils.rotx(link_info[2, 6]) * Utils.trn(
        [link_info[3, 6], 0, 0]))
    Q08 = Q07 * (Utils.rotz(link_info[0, 6] + theta8) * Utils.trn([0, 0, link_info[1, 7]]) * Utils.rotx(link_info[2, 7]) * Utils.trn(
        [link_info[3, 7], 0, 0]))
    Q09 = Q08 * (Utils.rotz(link_info[0, 7]) * Utils.trn([0, 0, link_info[1, 8]]) * Utils.rotx(link_info[2, 8]) * Utils.trn(
        [link_info[3, 8], 0, 0]))

    link1_mth = Utils.inv_htm(Q01)
    link_3d_obj.append([
        # torre movel
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/58.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
        # cilindro conector
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/63.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
    ])

    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        # ligacao de cilindros longa
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/64.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
    ])

    link3_mth = Utils.inv_htm(Q03)
    link_3d_obj.append([
        # ligacao de cilindros longa
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/61.obj',
                scale=scale, htm=link3_mth, mesh_material=mesh),
    ])

    link4_mth = Utils.inv_htm(Q04)
    link_3d_obj.append([
        # cotovelo
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/14.obj',
                scale=scale, htm=link4_mth, mesh_material=mesh),
    ])

    link5_mth = Utils.inv_htm(Q05)
    link_3d_obj.append([
        # pá dobrada
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/36.obj',
                scale=scale, htm=link5_mth, mesh_material=mesh),
    ])

    link6_mth = Utils.inv_htm(Q06)
    link_3d_obj.append([
        # conecta pá com bumerangue
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/34.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh),
    ])

    link7_mth = Utils.inv_htm(Q07)
    link_3d_obj.append([
        # bumerangue
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/39.obj',
                scale=scale, htm=link7_mth, mesh_material=mesh),
    ])

    link8_mth = Utils.inv_htm(Q08)
    link_3d_obj.append([
        # envoltoria agulha
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/42.obj',
                scale=scale, htm=link8_mth, mesh_material=mesh),
    ])

    link9_mth = Utils.inv_htm(Q09)
    link_3d_obj.append([
        # guia
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/23.obj',
                    scale=scale, htm=link9_mth, mesh_material=mesh),
        # agulha
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/55.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh3),
        # abridor de garrafa
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/25.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh),
        # botao
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/53.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh2),
    ])

    col_model = [[], [], [], [], [], [], [], [], []]

    col_model[0].append(Box(htm=Utils.trn([-0.02, 0.006*0, -0.43]) @ Utils.rotz(np.pi/2),
                            name=name + "_C0_0", width=0.14, height=0.4, depth=0.2, color="red", opacity=0.3))
    col_model[0].append(Cylinder(htm=Utils.trn([0, 0, -0.16]) @ Utils.rotz(3.14 / 2) @ Utils.rotx(3.14),
                                 name=name + "_C0_1", radius=0.075, height=0.19, color="red", opacity=0.3))

    col_model[1].append(Box(htm=Utils.trn([-0.200, 0, -0.11]) @ Utils.roty(np.pi/2),
                            name=name + "_C1_0", width=0.1, height=0.4, depth=0.12, color="green", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([0, 0, -0.12]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C1_1", radius=0.075, height=0.12, color="green", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([-0.415, 0, -0.11]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C1_2", radius=0.075, height=0.13, color="green", opacity=0.3))

    col_model[2].append(Box(htm=Utils.trn([-0.200, 0, 0.005]) @ Utils.roty(np.pi/2),
                            name=name + "_C2_0", width=0.1, height=0.4, depth=0.12, color="blue", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([0, 0, 0.01]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C2_1", radius=0.075, height=0.12, color="blue", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([-0.415, 0, 0.01]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C2_2", radius=0.075, height=0.13, color="blue", opacity=0.3))

    col_model[3].append(Cylinder(htm=Utils.trn([0, -0.025, 0.01]) @ Utils.rotx(-alpha4),
                                 name=name + "_C3_0", radius=0.075, height=0.12, color="magenta", opacity=0.3))
    col_model[3].append(Box(htm=Utils.trn([0, -0.02, 0.12]) @ Utils.roty(np.pi/2),
                            name=name + "_C3_1", width=0.2, height=0.14, depth=0.14, color="magenta", opacity=0.3))

    col_model[4].append(Box(htm=Utils.trn([0, -0.355, -0.05]) @ Utils.roty(np.pi/2),
                            name=name + "_C4_0", width=0.2, height=0.08, depth=0.05, color="orange", opacity=0.3))
    col_model[4].append(Box(htm=Utils.trn([0, -0.15, -0.125]) @ Utils.roty(np.pi/2),
                            name=name + "_C4_1", width=0.05, height=0.09, depth=0.4, color="orange", opacity=0.3))

    col_model[5].append(Box(htm=Utils.trn([-0.2, 0.025, -0.025]),
                            name=name + "_C5_0", width=0.25, height=0.14, depth=0.15, color="cyan", opacity=0.3))
    col_model[5].append(Box(htm=Utils.trn([-0.02, 0.025, -0.088]),
                            name=name + "_C5_1", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))
    col_model[5].append(Box(htm=Utils.trn([-0.019, 0.025, 0.019]),
                            name=name + "_C5_2", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))

    col_model[6].append(Box(htm=Utils.trn([-0.42, 0.08, -0.034]) @ Utils.rotz(np.pi/3),
                            name=name + "_C6_0", width=0.23, height=0.082, depth=0.13, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.22, 0.08, -0.034]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_1", width=0.11, height=0.09, depth=0.39, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.023, 0.01, 0.005]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_2", width=0.08, height=0.01, depth=0.03, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.023, 0.01, -0.068]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_3", width=0.08, height=0.01, depth=0.03, color="#88264a", opacity=0.3))
    col_model[6].append(Cylinder(htm=Utils.trn([0, 0, -0.068]),
                                 name=name + "_C6_4", radius=0.035, height=0.01, color="#88264a", opacity=0.3))
    col_model[6].append(Cylinder(htm=Utils.trn([0, 0, 0.005]),
                                 name=name + "_C6_5", radius=0.035, height=0.01, color="#88264a", opacity=0.3))

    col_model[7].append(Box(htm=Utils.trn([-0.052, 0.0, -0.042]),
                            name=name + "_C7_0", width=0.02, height=0.31, depth=0.12, color="Brown", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([-0.1, -0.005, -0.001]) @ Utils.rotx(np.pi/2),
                                 name=name + "_C7_1", radius=0.035, height=0.056, color="Brown", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.08, -0.005, 0]),
                            name=name + "_C7_2", width=0.04, height=0.07, depth=0.056, color="Brown", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.03, -0.001, 0.073]),
                            name=name + "_C7_3", width=0.04, height=0.05, depth=0.04, color="Brown", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([0, -0.001, 0.074]),
                                 name=name + "_C7_4", radius=0.02, height=0.056, color="Brown", opacity=0.3))

    col_model[8].append(Box(htm=Utils.trn([-0.005, 0.0, -0.22]),
                            name=name + "_C8_0", width=0.07, height=0.4, depth=0.12, color="MidnightBlue", opacity=0.3))
    col_model[8].append(Cylinder(htm=Utils.trn([0, 0, 0.135]),
                                 name=name + "_C8_1", radius=0.01, height=0.4, color="MidnightBlue", opacity=0.3))

    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i]))
        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    #     1  2  3  4  5  6  7  8  9
    q0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    htm_n_eef = Utils.trn([0, 0, 0.3323])  # np.identity(4)
    htm_base_0 = Utils.trn([-b1, -b2, b3])  # np.identity(4)

    # Create joint limits
    joint_limits = np.matrix([[-0.1, 0.53], [-4*np.pi, 4*np.pi], [-np.deg2rad(162), np.deg2rad(162)], [-4*np.pi, 4*np.pi], [-4*np.pi, 4*np.pi],
                             [-4*np.pi, 4*np.pi], [-np.deg2rad(175), np.deg2rad(50)], [-np.deg2rad(108), np.deg2rad(70)], [-0.05, 0.079]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits


def _create_davinci_arm3(color=["#919090", "#61b3c7", "#a6a6a6"], opacity=1, name='davinci_arm3'):
    color1, color2, color3 = color

    theta2 = np.deg2rad(-3)
    theta3 = np.deg2rad(98.5)
    theta4 = np.deg2rad(108.5)
    theta5 = np.deg2rad(-44)
    theta6 = np.deg2rad(-85.5)
    theta7 = np.deg2rad(-171)
    theta8 = np.deg2rad(61.7)

    d2 = 96e-3 * 1.2
    d4 = -96e-3 * 0.925
    d5 = 431.8e-3 * 1.419
    d8 = -2.9e-2

    alpha4 = np.deg2rad(90 + 36.5)
    alpha5 = np.pi/2
    alpha8 = np.deg2rad(90)

    a2 = 0.415
    a3 = 0.415
    a5 = -8e-4
    a6 = 0.27
    a7 = 0.474
    a8 = 0.098

    # Passive joints 1, 2, 3, 4, 7, 8
    # Active joints 5, 6, 9
    link_info = np.array([
        # "theta" rotation in z
        [0,  0,   0,       0,      0,      0,      0,       0, 0],
        # 0,  "d" translation in z
        [0, d2,   0,      d4,     d5,      0,      0,      d8, 0],
        # 0,  "alfa" rotation in x
        [0,  0,   0,  alpha4, alpha5,      0,      0,  alpha8, 0],
        # 0,  "a" translation in x
        [0, a2,  a3,       0,     a5,     a6,     a7,      a8, 0],
        # 0,  joint type
        [1,  0,   0,       0,      0,      0,      0,       0, 1]
    ])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    link_3d_obj = []
    mesh = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color1, opacity=opacity, side="DoubleSide")
    mesh2 = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color2, opacity=opacity, side="DoubleSide")
    mesh3 = MeshMaterial(metalness=0.8, roughness=0.4, clearcoat=0, normal_scale=[
                        0.5, 0.5], color=color3, opacity=opacity, side="DoubleSide")
    b1 = -0.086
    b2 = 0.2634
    b3 = 1.3

    Q00 = Utils.trn([b1, b3, -b2]) * Utils.rotx(-np.pi/2) * \
        Utils.rotz(np.pi)  # change reference frame
    Q01 = Q00 * (Utils.rotz(link_info[0, 0]) * Utils.trn([0, 0, link_info[1, 0]]) * Utils.rotx(link_info[2, 0]) * Utils.trn(
        [link_info[3, 0], 0, 0]))
    Q02 = Q01 * (Utils.rotz(link_info[0, 0] + theta2) * Utils.trn([0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn(
        [link_info[3, 1], 0, 0]))
    Q03 = Q02 * (Utils.rotz(link_info[0, 1] + theta3) * Utils.trn([0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn(
        [link_info[3, 2], 0, 0]))
    Q04 = Q03 * (Utils.rotz(link_info[0, 2] + theta4) * Utils.trn([0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn(
        [link_info[3, 3], 0, 0]))
    Q05 = Q04 * (Utils.rotz(link_info[0, 3] + theta5) * Utils.trn([0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn(
        [link_info[3, 4], 0, 0]))
    Q06 = Q05 * (Utils.rotz(link_info[0, 4] + theta6) * Utils.trn([0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn(
        [link_info[3, 5], 0, 0]))
    Q07 = Q06 * (Utils.rotz(link_info[0, 5] + theta7) * Utils.trn([0, 0, link_info[1, 6]]) * Utils.rotx(link_info[2, 6]) * Utils.trn(
        [link_info[3, 6], 0, 0]))
    Q08 = Q07 * (Utils.rotz(link_info[0, 6] + theta8) * Utils.trn([0, 0, link_info[1, 7]]) * Utils.rotx(link_info[2, 7]) * Utils.trn(
        [link_info[3, 7], 0, 0]))
    Q09 = Q08 * (Utils.rotz(link_info[0, 7]) * Utils.trn([0, 0, link_info[1, 8]]) * Utils.rotx(link_info[2, 8]) * Utils.trn(
        [link_info[3, 8], 0, 0]))

    link1_mth = Utils.inv_htm(Q01)
    link_3d_obj.append([
        # torre movel
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/58.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
        # cilindro conector
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/63.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
    ])

    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        # ligacao de cilindros longa
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/64.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
    ])

    link3_mth = Utils.inv_htm(Q03)
    link_3d_obj.append([
        # ligacao de cilindros longa
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/61.obj',
                scale=scale, htm=link3_mth, mesh_material=mesh),
    ])

    link4_mth = Utils.inv_htm(Q04)
    link_3d_obj.append([
        # cotovelo
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/14.obj',
                scale=scale, htm=link4_mth, mesh_material=mesh),
    ])

    link5_mth = Utils.inv_htm(Q05)
    link_3d_obj.append([
        # pá dobrada
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/36.obj',
                scale=scale, htm=link5_mth, mesh_material=mesh),
    ])

    link6_mth = Utils.inv_htm(Q06)
    link_3d_obj.append([
        # conecta pá com bumerangue
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/34.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh),
    ])

    link7_mth = Utils.inv_htm(Q07)
    link_3d_obj.append([
        # bumerangue
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/39.obj',
                scale=scale, htm=link7_mth, mesh_material=mesh),
    ])

    link8_mth = Utils.inv_htm(Q08)
    link_3d_obj.append([
        # envoltoria agulha
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/42.obj',
                scale=scale, htm=link8_mth, mesh_material=mesh),
    ])

    link9_mth = Utils.inv_htm(Q09)
    link_3d_obj.append([
        # guia
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/23.obj',
                    scale=scale, htm=link9_mth, mesh_material=mesh),
        # agulha
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/55.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh3),
        # abridor de garrafa
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/25.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh),
        # botao
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/53.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh2),
    ])

    col_model = [[], [], [], [], [], [], [], [], []]

    col_model[0].append(Box(htm=Utils.trn([-0.02, 0.006*0, -0.43]) @ Utils.rotz(np.pi/2),
                            name=name + "_C0_0", width=0.14, height=0.4, depth=0.2, color="red", opacity=0.3))
    col_model[0].append(Cylinder(htm=Utils.trn([0, 0, -0.16]) @ Utils.rotz(3.14 / 2) @ Utils.rotx(3.14),
                                 name=name + "_C0_1", radius=0.075, height=0.19, color="red", opacity=0.3))

    col_model[1].append(Box(htm=Utils.trn([-0.200, 0, -0.11]) @ Utils.roty(np.pi/2),
                            name=name + "_C1_0", width=0.1, height=0.4, depth=0.12, color="green", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([0, 0, -0.12]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C1_1", radius=0.075, height=0.12, color="green", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([-0.415, 0, -0.11]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C1_2", radius=0.075, height=0.13, color="green", opacity=0.3))

    col_model[2].append(Box(htm=Utils.trn([-0.200, 0, 0.005]) @ Utils.roty(np.pi/2),
                            name=name + "_C2_0", width=0.1, height=0.4, depth=0.12, color="blue", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([0, 0, 0.01]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C2_1", radius=0.075, height=0.12, color="blue", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([-0.415, 0, 0.01]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C2_2", radius=0.075, height=0.13, color="blue", opacity=0.3))

    col_model[3].append(Cylinder(htm=Utils.trn([0, -0.025, 0.01]) @ Utils.rotx(-alpha4),
                                 name=name + "_C3_0", radius=0.075, height=0.12, color="magenta", opacity=0.3))
    col_model[3].append(Box(htm=Utils.trn([0, -0.02, 0.12]) @ Utils.roty(np.pi/2),
                            name=name + "_C3_1", width=0.2, height=0.14, depth=0.14, color="magenta", opacity=0.3))

    col_model[4].append(Box(htm=Utils.trn([0, -0.355, -0.05]) @ Utils.roty(np.pi/2),
                            name=name + "_C4_0", width=0.2, height=0.08, depth=0.05, color="orange", opacity=0.3))
    col_model[4].append(Box(htm=Utils.trn([0, -0.15, -0.125]) @ Utils.roty(np.pi/2),
                            name=name + "_C4_1", width=0.05, height=0.09, depth=0.4, color="orange", opacity=0.3))

    col_model[5].append(Box(htm=Utils.trn([-0.2, 0.025, -0.025]),
                            name=name + "_C5_0", width=0.25, height=0.14, depth=0.15, color="cyan", opacity=0.3))
    col_model[5].append(Box(htm=Utils.trn([-0.02, 0.025, -0.088]),
                            name=name + "_C5_1", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))
    col_model[5].append(Box(htm=Utils.trn([-0.019, 0.025, 0.019]),
                            name=name + "_C5_2", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))

    col_model[6].append(Box(htm=Utils.trn([-0.42, 0.08, -0.034]) @ Utils.rotz(np.pi/3),
                            name=name + "_C6_0", width=0.23, height=0.082, depth=0.13, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.22, 0.08, -0.034]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_1", width=0.11, height=0.09, depth=0.39, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.023, 0.01, 0.005]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_2", width=0.08, height=0.01, depth=0.03, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.023, 0.01, -0.068]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_3", width=0.08, height=0.01, depth=0.03, color="#88264a", opacity=0.3))
    col_model[6].append(Cylinder(htm=Utils.trn([0, 0, -0.068]),
                                 name=name + "_C6_4", radius=0.035, height=0.01, color="#88264a", opacity=0.3))
    col_model[6].append(Cylinder(htm=Utils.trn([0, 0, 0.005]),
                                 name=name + "_C6_5", radius=0.035, height=0.01, color="#88264a", opacity=0.3))

    col_model[7].append(Box(htm=Utils.trn([-0.052, 0.0, -0.042]),
                            name=name + "_C7_0", width=0.02, height=0.31, depth=0.12, color="Brown", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([-0.1, -0.005, -0.001]) @ Utils.rotx(np.pi/2),
                                 name=name + "_C7_1", radius=0.035, height=0.056, color="Brown", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.08, -0.005, 0]),
                            name=name + "_C7_2", width=0.04, height=0.07, depth=0.056, color="Brown", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.03, -0.001, 0.073]),
                            name=name + "_C7_3", width=0.04, height=0.05, depth=0.04, color="Brown", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([0, -0.001, 0.074]),
                                 name=name + "_C7_4", radius=0.02, height=0.056, color="Brown", opacity=0.3))

    col_model[8].append(Box(htm=Utils.trn([-0.005, 0.0, -0.22]),
                            name=name + "_C8_0", width=0.07, height=0.4, depth=0.12, color="MidnightBlue", opacity=0.3))
    col_model[8].append(Cylinder(htm=Utils.trn([0, 0, 0.135]),
                                 name=name + "_C8_1", radius=0.01, height=0.4, color="MidnightBlue", opacity=0.3))

    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i]))
        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    #     1  2  3  4  5  6  7  8  9
    q0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    htm_n_eef = Utils.trn([0, 0, 0.3323])  # np.identity(4)
    htm_base_0 = Utils.rotz(
        np.pi) * Utils.trn([-3.4*b1, b2, b3])  # np.identity(4)

    # Create joint limits
    joint_limits = np.matrix([[-0.1, 0.53], [-np.pi, np.pi], [-np.deg2rad(162), np.deg2rad(162)], [-np.pi, np.pi], [-np.pi, np.pi],
                             [-np.pi, np.pi], [-np.deg2rad(175), np.deg2rad(50)], [-np.deg2rad(108), np.deg2rad(70)], [-0.05, 0.079]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits


def _create_davinci_arm4(color=["#919090", "#61b3c7", "#a6a6a6"], opacity=1, name='davinci_arm4'):
    color1, color2, color3 = color

    theta2 = np.deg2rad(164.8)
    theta3 = np.deg2rad(-131.8)
    theta4 = np.deg2rad(-33)
    theta5 = np.deg2rad(-91)
    theta6 = np.deg2rad(94.5)
    theta7 = np.deg2rad(-169.82)
    theta8 = np.deg2rad(37.1)

    d3 = 96e-3 * 1.25
    d4 = 96e-3 * 3.68
    d5 = 0.919
    d8 = -3.4e-2
    d9 = 0.069

    alpha4 = -np.deg2rad(90 + 63.1)
    alpha5 = -np.pi/2
    alpha8 = np.deg2rad(90.7)

    a2 = 0.3
    a3 = 0.3
    a5 = -8e-4
    a6 = 0.27
    a7 = 0.473
    a8 = 0.115

    # Passive joints 1, 2, 3, 4, 7, 8
    # Active joints 5, 6, 9
    link_info = np.array([
        # "theta" rotation in z
        [0,  0,   0,       0,      0,      0,      0,       0, 0],
        # 0,  "d" translation in z
        [0,  0,  d3,      d4,     d5,      0,      0,      d8, d9],
        # 0,  "alfa" rotation in x
        [0,  0,   0,  alpha4, alpha5,      0,      0,  alpha8, 0],
        # 0,  "a" translation in x
        [0, a2,  a3,       0,     a5,     a6,     a7,      a8, 0],
        # 0,  joint type
        [1,  0,   0,       0,      0,      0,      0,       0, 1]
    ])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    link_3d_obj = []
    mesh = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color1, opacity=opacity, side="DoubleSide")
    mesh2 = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color2, opacity=opacity, side="DoubleSide")
    mesh3 = MeshMaterial(metalness=0.8, roughness=0.4, clearcoat=0, normal_scale=[
                        0.5, 0.5], color=color3, opacity=opacity, side="DoubleSide")
    b1 = 0.099
    b2 = 0.0922
    b3 = 1.54

    Q00 = Utils.trn([b1, b3, -b2]) * Utils.rotx(-np.pi/2) * \
        Utils.rotz(np.pi)  # change reference frame
    Q01 = Q00 * (Utils.rotz(link_info[0, 0]) * Utils.trn([0, 0, link_info[1, 0]]) * Utils.rotx(link_info[2, 0]) * Utils.trn(
        [link_info[3, 0], 0, 0]))
    Q02 = Q01 * (Utils.rotz(link_info[0, 0] + theta2) * Utils.trn([0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn(
        [link_info[3, 1], 0, 0]))
    Q03 = Q02 * (Utils.rotz(link_info[0, 1] + theta3) * Utils.trn([0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn(
        [link_info[3, 2], 0, 0]))
    Q04 = Q03 * (Utils.rotz(link_info[0, 2] + theta4) * Utils.trn([0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn(
        [link_info[3, 3], 0, 0]))
    Q05 = Q04 * (Utils.rotz(link_info[0, 3] + theta5) * Utils.trn([0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn(
        [link_info[3, 4], 0, 0]))
    Q06 = Q05 * (Utils.rotz(link_info[0, 4] + theta6) * Utils.trn([0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn(
        [link_info[3, 5], 0, 0]))
    Q07 = Q06 * (Utils.rotz(link_info[0, 5] + theta7) * Utils.trn([0, 0, link_info[1, 6]]) * Utils.rotx(link_info[2, 6]) * Utils.trn(
        [link_info[3, 6], 0, 0]))
    Q08 = Q07 * (Utils.rotz(link_info[0, 6] + theta8) * Utils.trn([0, 0, link_info[1, 7]]) * Utils.rotx(link_info[2, 7]) * Utils.trn(
        [link_info[3, 7], 0, 0]))
    Q09 = Q08 * (Utils.rotz(link_info[0, 7]) * Utils.trn([0, 0, link_info[1, 8]]) * Utils.rotx(link_info[2, 8]) * Utils.trn(
        [link_info[3, 8], 0, 0]))

    link1_mth = Utils.inv_htm(Q01)
    link_3d_obj.append([
        # torre movel
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/57.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
        # cilindro conector
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/60.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
    ])

    link2_mth = Utils.inv_htm(Q02)
    link_3d_obj.append([
        # ligacao de cilindros curta
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/15.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
    ])

    link3_mth = Utils.inv_htm(Q03)
    link_3d_obj.append([
        # ligacao de cilindros curta
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/66.obj',
                scale=scale, htm=link3_mth, mesh_material=mesh),
    ])

    link4_mth = Utils.inv_htm(Q04)
    link_3d_obj.append([
        # cotovelo
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/16.obj',
                scale=scale, htm=link4_mth, mesh_material=mesh),
    ])

    link5_mth = Utils.inv_htm(Q05)
    link_3d_obj.append([
        # pá dobrada
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/37.obj',
                scale=scale, htm=link5_mth, mesh_material=mesh),
    ])

    link6_mth = Utils.inv_htm(Q06)
    link_3d_obj.append([
        # conecta pá com bumerangue
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/33.obj',
                scale=scale, htm=link6_mth, mesh_material=mesh),
    ])

    link7_mth = Utils.inv_htm(Q07)
    link_3d_obj.append([
        # bumerangue
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/41.obj',
                scale=scale, htm=link7_mth, mesh_material=mesh),
    ])

    link8_mth = Utils.inv_htm(Q08)
    link_3d_obj.append([
        # envoltoria agulha
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/28.obj',
                scale=scale, htm=link8_mth, mesh_material=mesh),
    ])

    link9_mth = Utils.inv_htm(Q09)
    link_3d_obj.append([
        # guia
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/29.obj',
                    scale=scale, htm=link9_mth, mesh_material=mesh),
        # agulha
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/31.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh3),
        # base agulha
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/30.obj',
                scale=scale, htm=link9_mth, mesh_material=mesh),
        # cabo longo
        # Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/32.obj',
        #        scale=scale, htm=link9_mth, mesh_material=mesh),
    ])

    col_model = [[], [], [], [], [], [], [], [], []]

    col_model[0].append(Box(htm=Utils.trn([0, -0.03, -0.33]),
                            name=name + "_C0_0", width=0.14, height=0.4, depth=0.2, color="red", opacity=0.3))
    col_model[0].append(Cylinder(htm=Utils.trn([0, 0, -0.154]) @ Utils.rotz(3.14 / 2) @ Utils.rotx(3.14),
                                 name=name + "_C0_1", radius=0.075, height=0.19, color="red", opacity=0.3))

    col_model[1].append(Box(htm=Utils.trn([-0.15, 0, 0]) @ Utils.roty(np.pi/2),
                            name=name + "_C1_0", width=0.11, height=0.3, depth=0.12, color="green", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([0, 0, -0.001]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C1_1", radius=0.075, height=0.123, color="green", opacity=0.3))
    col_model[1].append(Cylinder(htm=Utils.trn([-0.3, 0, 0.002]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C1_2", radius=0.075, height=0.12, color="green", opacity=0.3))

    col_model[2].append(Box(htm=Utils.trn([-0.15, 0, 0.0]) @ Utils.roty(np.pi/2),
                            name=name + "_C2_0", width=0.11, height=0.3, depth=0.12, color="blue", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([0, 0, -0.025]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C2_1", radius=0.075, height=0.08, color="blue", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([-0.3, 0, 0.002]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C2_2", radius=0.075, height=0.12, color="blue", opacity=0.3))
    col_model[2].append(Cylinder(htm=Utils.trn([0, 0, 0.033]) @ Utils.rotz(np.pi/2) @ Utils.rotx(3.14),
                                 name=name + "_C2_3", radius=0.062, height=0.05, color="blue", opacity=0.3))

    col_model[3].append(Cylinder(htm=Utils.rotx(-alpha4) @ Utils.trn([0, 0, 0.101-d4]),
                                 name=name + "_C3_0", radius=0.075, height=0.08, color="magenta", opacity=0.3))
    col_model[3].append(Box(htm=Utils.trn([0, 0.02, 0.42]) @ Utils.roty(np.pi/2),
                            name=name + "_C3_1", width=0.23, height=0.14, depth=0.12, color="magenta", opacity=0.3))
    col_model[3].append(Box(htm=Utils.rotx(-alpha4) @ Utils.trn([0, 0.072, -0.259]),
                            name=name + "_C3_2", width=0.14, height=0.07, depth=0.2, color="magenta", opacity=0.3))

    col_model[4].append(Box(htm=Utils.trn([0, 0.355, -0.05]) @ Utils.roty(np.pi/2),
                            name=name + "_C4_0", width=0.2, height=0.08, depth=0.05, color="orange", opacity=0.3))
    col_model[4].append(Box(htm=Utils.trn([0, 0.15, -0.1288]) @ Utils.roty(np.pi/2),
                            name=name + "_C4_1", width=0.05, height=0.09, depth=0.4, color="orange", opacity=0.3))

    col_model[5].append(Box(htm=Utils.trn([-0.2, 0.025, -0.03]),
                            name=name + "_C5_0", width=0.25, height=0.14, depth=0.15, color="cyan", opacity=0.3))
    col_model[5].append(Box(htm=Utils.trn([-0.018, 0.02, -0.09]),
                            name=name + "_C5_1", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))
    col_model[5].append(Box(htm=Utils.trn([-0.019, 0.02, 0.015]),
                            name=name + "_C5_2", width=0.14, height=0.02, depth=0.15, color="cyan", opacity=0.3))

    col_model[6].append(Box(htm=Utils.trn([-0.42, 0.08, -0.0372]) @ Utils.rotz(np.pi/3),
                            name=name + "_C6_0", width=0.23, height=0.082, depth=0.13, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.22, 0.08, -0.034]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_1", width=0.11, height=0.09, depth=0.39, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.023, 0.008, 0.002]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_2", width=0.08, height=0.01, depth=0.03, color="#88264a", opacity=0.3))
    col_model[6].append(Box(htm=Utils.trn([-0.023, 0.008, -0.071]) @ Utils.rotz(np.pi*7/18),
                            name=name + "_C6_3", width=0.08, height=0.01, depth=0.03, color="#88264a", opacity=0.3))
    col_model[6].append(Cylinder(htm=Utils.trn([0, 0.001, -0.071]),
                                 name=name + "_C6_4", radius=0.035, height=0.01, color="#88264a", opacity=0.3))
    col_model[6].append(Cylinder(htm=Utils.trn([0, 0, 0.002]),
                                 name=name + "_C6_5", radius=0.035, height=0.01, color="#88264a", opacity=0.3))

    col_model[7].append(Box(htm=Utils.trn([-0.065, 0, -0.18]),
                            name=name + "_C7_0", width=0.028, height=0.63, depth=0.12, color="Brown", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([-0.115, 0.001, -0.0013]) @ Utils.rotx(np.pi/2),
                                 name=name + "_C7_1", radius=0.035, height=0.056, color="Brown", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.09, -0.005*0, 0.0]),
                            name=name + "_C7_2", width=0.04, height=0.07, depth=0.056, color="Brown", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.031, -0.001*0, 0.095]),
                            name=name + "_C7_3", width=0.048, height=0.09, depth=0.04, color="Brown", opacity=0.3))
    col_model[7].append(Cylinder(htm=Utils.trn([0.01, 0.001*0, 0.112]),
                                 name=name + "_C7_4", radius=0.025, height=0.045, color="Brown", opacity=0.3))
    col_model[7].append(Box(htm=Utils.trn([-0.1, 0, -0.306]),
                            name=name + "_C7_5", width=0.04, height=0.378, depth=0.12, color="Brown", opacity=0.3))

    col_model[8].append(Box(htm=Utils.trn([-0.002, 0.0, -0.52]),
                            name=name + "_C8_0", width=0.09, height=0.36, depth=0.12, color="MidnightBlue", opacity=0.3))
    col_model[8].append(Cylinder(htm=Utils.trn([0, 0, -0.109]),
                                 name=name + "_C8_1", radius=0.013, height=0.6, color="MidnightBlue", opacity=0.3))

    links = []
    for i in range(n):
        links.append(Link(i, theta=link_info[0, i], d=link_info[1, i], alpha=link_info[2, i], a=link_info[3, i], joint_type=link_info[4, i],
                          list_model_3d=link_3d_obj[i]))
        for j in range(len(col_model[i])):
            links[i].attach_col_object(col_model[i][j], col_model[i][j].htm)

    # Define initial configuration
    #     1  2  3  4  5  6  7  8  9
    q0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    htm_n_eef = Utils.trn([0, 0, 0.1899 - d9])  # np.identity(4)
    htm_base_0 = Utils.trn([-b1, -b2, b3])  # np.identity(4)

    # Create joint limits
    joint_limits = np.matrix([[-0.423, 0.21], [-np.pi, np.pi], [-np.deg2rad(158), np.deg2rad(156)], [-np.deg2rad(200), np.deg2rad(
        20)], [-np.pi, np.pi], [-np.pi, np.pi], [-np.deg2rad(170), np.deg2rad(50)], [-np.deg2rad(108), np.deg2rad(60)], [0, 0.295]])

    return links, base_3d_obj, htm_base_0, htm_n_eef, q0, joint_limits