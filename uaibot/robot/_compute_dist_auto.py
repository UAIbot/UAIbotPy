from uaibot.utils import *
import numpy as np
import math
from ._dist_struct_robot_auto import DistStructRobotAuto
from ._dist_struct_robot_auto import DistStructLinkLink
import os
if os.environ['CPP_SO_FOUND']=="1":
    import uaibot_cpp_bind as ub_cpp

def _diststructlinklink_cpp2py(dsll_cpp):

    dsll_py = DistStructLinkLink(dsll_cpp.link_number_1, dsll_cpp.link_col_obj_number_1, dsll_cpp.link_number_2, dsll_cpp.link_col_obj_number_2, dsll_cpp.distance, np.matrix(dsll_cpp.point_link_1).reshape((3,1)), np.matrix(dsll_cpp.point_link_2).reshape((3,1)), np.matrix(dsll_cpp.jac_distance))
    return dsll_py

def _diststructlinklink_py2cpp(dsll_py):

    dsll_cpp = ub_cpp.CPP_DistStructLinkLink()

    dsll_cpp.is_null = False
    dsll_cpp.link_number_1 = dsll_py.link_number_1
    dsll_cpp.link_number_2 = dsll_py.link_number_2
    dsll_cpp.link_col_obj_number_1 = dsll_py.link_col_obj_number_1
    dsll_cpp.link_col_obj_number_2 = dsll_py.link_col_obj_number_2
    dsll_cpp.distance = dsll_py.distance
    dsll_cpp.point_link_1 = dsll_py.point_link_1
    dsll_cpp.point_link_2 = dsll_py.point_link_2
    dsll_cpp.jac_distance = dsll_py.jac_distance

    return dsll_cpp

def _diststructrobotauto_cpp2py(dsra_cpp, _robot):

    dsra_py = DistStructRobotAuto(_robot)

    dsra_py._jac_dist_mat = np.matrix(dsra_cpp.jac_dist_mat)
    n = np.shape(dsra_cpp.dist_vect)[0]
    dsra_py._dist_vect = np.matrix(dsra_cpp.dist_vect).reshape((n,1))
    
    dsra_py._list_info = [_diststructlinklink_cpp2py(dsll) for dsll in dsra_cpp.list_info]
    dsra_py._no_items = len(dsra_py._list_info)

    return dsra_py

def _diststructrobotauto_py2cpp(dsra_py):

    dsra_cpp = ub_cpp.CPP_DistStructRobotAuto()

    dsra_cpp.is_null = dsra_py is None
    if not (dsra_py is None):
        dsra_cpp.jac_dist_mat = dsra_py.jac_dist_mat
        dsra_cpp.dist_vect = dsra_py.dist_vect
        dsra_cpp.list_info = [_diststructlinklink_py2cpp(dsll) for dsll in dsra_py._list_info]

    return dsra_cpp


def _compute_dist_auto(self, q=None, old_dist_struct=None, tol=0.0005, no_iter_max=20, max_dist = np.inf, h=0, eps = 0, mode='auto'):
    n = len(self.links)

    if q is None:
        q = self.q

    # Error handling
    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not Utils.is_a_number(tol) or tol <= 0:
        raise Exception("The parameter 'tol' should be a positive number.")

    if not Utils.is_a_number(max_dist) or tol <= 0:
        raise Exception("The parameter 'max_dist' should be a positive number, or np.inf.")

    if not Utils.is_a_natural_number(no_iter_max) or no_iter_max <= 0:
        raise Exception("The parameter 'no_iter_max' should be a positive natural number.")

    if not (old_dist_struct is None):
        try:
            if not id(old_dist_struct.robot) == id(self):
                Exception("The parameter 'old_dist_struct' is a 'DistStructRobotAuto' object, but it " \
                          "must have to be relative to the SAME robot object, and " \
                          "this is not the case.")
        except:
            raise Exception("The parameter 'old_dist_struct' must be a 'DistStructRobotAuto' object.")

    if not Utils.is_a_number(h) or h < 0:
        raise Exception("The optional parameter 'h' must be a nonnegative number.")

    if not Utils.is_a_number(eps) or eps < 0:
        raise Exception("The optional parameter 'eps' must be a nonnegative number.")
    
    if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
        raise Exception("c++ mode is set, but .so file was not loaded!")
    # end error handling

    if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
        if h >0 or eps > 0:
            raise Exception("In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!")
         
        return _compute_dist_auto_python(self, Utils.cvt(q), old_dist_struct, tol, no_iter_max, max_dist)
    else:
        

        old_dsra = _diststructrobotauto_py2cpp(old_dist_struct)       
        new_dsra = self.cpp_robot.compute_dist_auto(Utils.cvt(q), old_dsra, tol, no_iter_max, max_dist, h+1e-6, eps+1e-6)

        return _diststructrobotauto_cpp2py(new_dsra, self)


def _compute_dist_auto_python(self, q=None, old_dist_struct=None, tol=0.0005, no_iter_max=20, max_dist = np.inf):
    n = len(self.links)

    dist_struct = DistStructRobotAuto(self)

    jac_dh, mth_dh = self.jac_geo(q, "dh", mode = "python")

    col_object_copy = []

    # Update all collision objects of all links
    for i in range(n):
        col_object_copy.append([])
        for j in range(len(self.links[i].col_objects)):
            temp_copy = self.links[i].col_objects[j][0].copy()
            htmd = self.links[i].col_objects[j][1]
            temp_copy.set_ani_frame(mth_dh[i][:, :] * htmd)
            col_object_copy[i].append(temp_copy)

    # Compute the distance structure
    for i in range(n):
        for j in range(i+2,n):
            for isub in range(len(self.links[i].col_objects)):
                for jsub in range(len(self.links[j].col_objects)):

                    est_dist = 0 if math.isinf(max_dist) else Utils.compute_aabbdist(col_object_copy[i][isub], col_object_copy[j][jsub])

                    if est_dist <= max_dist:

                        if old_dist_struct is None:
                            p_obj_0 = np.matrix(np.random.uniform(-100, 100, size=(3, 1)))
                        else:
                            p_obj_0 = old_dist_struct.get_item(i, isub, j, jsub).point_object

                        p_obj_i, p_obj_j, d = Utils.compute_dist(col_object_copy[i][isub], col_object_copy[j][jsub] \
                                                                 , p_obj_0, tol, no_iter_max)

                        jac_obj_i = jac_dh[i][0:3, :] - Utils.S(p_obj_i - mth_dh[i][0:3, 3]) * jac_dh[i][3:6, :]
                        jac_obj_j = jac_dh[j][0:3, :] - Utils.S(p_obj_j - mth_dh[j][0:3, 3]) * jac_dh[j][3:6, :]

                        jac_dist = (p_obj_i - p_obj_j).T * (jac_obj_i - jac_obj_j) / d
                        dist_struct._append(i, isub, j, jsub, d, p_obj_i, p_obj_j, jac_dist)



    return dist_struct
