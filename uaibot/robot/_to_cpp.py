import numpy as np
from utils import *
import uaibot_cpp_bind as ub_cpp



def _to_cpp(robot):

    n = len(robot.links)

    cpp_robot = ub_cpp.CPP_Manipulator(n)

    cpp_robot.set_htm_extra(robot.htm_base_0, robot.htm_n_eef)
    cpp_robot.gravity_vector = robot.gravity_vector

    for i in range(n):
        cpp_robot.set_joint_param(i, robot.links[i].theta, robot.links[i].d, robot.links[i].alpha, robot.links[i].a, int(robot.links[i].joint_type), robot.joint_limit[i,0], robot.joint_limit[i,1])
        cpp_robot.set_inertial_param(i, robot.links[i].mass, robot.links[i].center_of_mass, robot.links[i].inertia_tensor)
        for j in range(len(robot.links[i].col_objects)):
            prim = robot.links[i].col_objects[j][0]
            htm = robot.links[i].col_objects[j][1]
            cpp_prim = Utils.obj_to_cpp(prim, htm)            
            cpp_robot.add_geo_prim(i, cpp_prim)
        
    
    return cpp_robot







