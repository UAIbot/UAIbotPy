from utils import Utils

from graphics.meshmaterial import MeshMaterial
from graphics.model3d import Model3D
from simobjects.rigidobject import RigidObject

import numpy as np


def _create_davinci_chest(name="davinci_chest", color=["#707070", "#545353"], opacity=1):

    color1, color2 = color

    link_info = np.array([
        # "theta" rotation in z
        [0, 0, 0, 0,        0,       0, 0],
        # "d" translation in z
        [0, 0, 0, 0,        0,       0, 0],
        # "alfa" rotation in x
        [0, 0, 0, 0, -np.pi/2, np.pi/2, 0],
        # "a" translation in x
        [0, 0, 0, 0,        0,       0, 0],
        # joint type
        [0, 0, 0, 0,        0,       1, 1]
    ])

    scale = 1
    n = link_info.shape[1]
    base_3d_obj = []
    chest_obj = []
    mesh = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color1, opacity=opacity, side="DoubleSide")
    mesh2 = MeshMaterial(metalness=0, roughness=1, clearcoat=0, clearcoat_roughness=0.03, ior=1.45, normal_scale=[
                        0.5, 0.5], color=color2, opacity=opacity, side="DoubleSide")
    # original model is rotated (Robot fron = plane X x Y)

    Q01 = Utils.rotx(-np.pi/2) * Utils.rotz(np.pi) * Utils.rotz(link_info[0, 0]) * Utils.trn([0, 0, link_info[1, 0]]) * Utils.rotx(link_info[2, 0]) * Utils.trn(
        [link_info[3, 0], 0, 0])
    Q12 = Q01 * (Utils.rotz(link_info[0, 1]) * Utils.trn([0, 0, link_info[1, 1]]) * Utils.rotx(link_info[2, 1]) * Utils.trn(
        [link_info[3, 1], 0, 0]))
    Q23 = Q12 * (Utils.rotz(link_info[0, 2]) * Utils.trn([0, 0, link_info[1, 2]]) * Utils.rotx(link_info[2, 2]) * Utils.trn(
        [link_info[3, 2], 0, 0]))
    Q34 = Q23 * (Utils.rotz(link_info[0, 3]) * Utils.trn([0, 0, link_info[1, 3]]) * Utils.rotx(link_info[2, 3]) * Utils.trn(
        [link_info[3, 3], 0, 0]))
    Q45 = Q34 * (Utils.rotz(link_info[0, 4]) * Utils.trn([0, 0, link_info[1, 4]]) * Utils.rotx(link_info[2, 4]) * Utils.trn(
        [link_info[3, 4], 0, 0]))
    Q56 = Q45 * (Utils.rotz(link_info[0, 5]) * Utils.trn([0, 0, link_info[1, 5]]) * Utils.rotx(link_info[2, 5]) * Utils.trn(
        [link_info[3, 5], 0, 0]))
    Q67 = Q56 * (Utils.rotz(link_info[0, 6]) * Utils.trn([0, 0, link_info[1, 6]]) * Utils.rotx(link_info[2, 6]) * Utils.trn(
        [link_info[3, 6], 0, 0]))

    link1_mth = Utils.inv_htm(Q01)
    chest_obj.extend([
        # feet
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/1.obj',
                scale=scale, htm=link1_mth, mesh_material=mesh),
    ])

    link2_mth = Utils.inv_htm(Q12)
    chest_obj.extend([
        # base rectangle
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/2.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh2),
        # vertical tower
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/3.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/17.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
        # short cable
        Model3D(url='https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/DaVinci/18.obj',
                scale=scale, htm=link2_mth, mesh_material=mesh),
    ])

    chest = RigidObject(chest_obj, name + "_chest")

    return chest
