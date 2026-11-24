from uaibot.utils import *
import numpy as np
import math
from ._dist_struct_robot_obj import DistStructRobotObj
from ._dist_struct_robot_obj import DistStructLinkObj
import os
if os.environ['CPP_SO_FOUND']=="1":
    import uaibot_cpp_bind as ub_cpp

# Compute the distance from each link to an object, for the current configuration
# of the robot


def _diststructlinkobj_cpp2py(dslo_cpp):

    dslo_py = DistStructLinkObj(dslo_cpp.link_number, dslo_cpp.link_col_obj_number, dslo_cpp.distance, np.matrix(dslo_cpp.point_link).reshape((3,1)), np.matrix(dslo_cpp.point_object).reshape((3,1)), np.matrix(dslo_cpp.jac_distance))

    #dslo_py = DistStructLinkObj(dslo_cpp.link_number, dslo_cpp.link_col_obj_number, dslo_cpp.distance, dslo_cpp.point_link, dslo_cpp.point_object, dslo_cpp.jac_distance)


    return dslo_py

def _diststructlinkobj_py2cpp(dslo_py):

    dslo_cpp = ub_cpp.CPP_DistStructLinkObj()

    dslo_cpp.is_null = False
    dslo_cpp.link_number = dslo_py.link_number
    dslo_cpp.link_col_obj_number = dslo_py.link_col_obj_number
    dslo_cpp.distance = dslo_py.distance
    dslo_cpp.point_link = dslo_py.point_link
    dslo_cpp.point_object = dslo_py.point_object
    dslo_cpp.jac_distance = dslo_py.jac_distance

    return dslo_cpp

def _diststructrobotobj_cpp2py(dslo_cpp, _obj, _robot):

    dsro_py = DistStructRobotObj(_obj, _robot)

    dsro_py._jac_dist_mat = np.matrix(dslo_cpp.jac_dist_mat)
    n = np.shape(dslo_cpp.dist_vect)[0]
    dsro_py._dist_vect = np.matrix(dslo_cpp.dist_vect).reshape((n,1))
    
    dsro_py._list_info = [_diststructlinkobj_cpp2py(dslo) for dslo in dslo_cpp.list_info]
    dsro_py._no_items = len(dsro_py._list_info)

    return dsro_py

def _diststructrobotobj_py2cpp(dslo_py):

    dsro_cpp = ub_cpp.CPP_DistStructRobotObj()

    dsro_cpp.is_null = dslo_py is None
    if not (dslo_py is None):
        dsro_cpp.jac_dist_mat = dslo_py.jac_dist_mat
        dsro_cpp.dist_vect = dslo_py.dist_vect
        dsro_cpp.list_info = [_diststructlinkobj_py2cpp(dslo) for dslo in dslo_py._list_info]

    return dsro_cpp



def _compute_dist(self, obj, q=None, htm=None, old_dist_struct=None, tol=0.0005, no_iter_max=20,
                  max_dist = np.inf, h=0, eps = 0, mode='auto'):
    
    n = len(self.links)

    if q is None:
        q = self.q

    if htm is None:
        htm = self.htm

    # Error handling
    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not Utils.is_a_metric_object(obj):
        raise Exception("The parameter 'obj' should be one of the following types: " + str(Utils.IS_METRIC) + ".")

    if not Utils.is_a_number(tol) or tol <= 0:
        raise Exception("The parameter 'tol' should be a positive number.")

    if not Utils.is_a_number(max_dist) or tol <= 0:
        raise Exception("The parameter 'max_dist' should be a positive number, or np.inf.")

    if not Utils.is_a_natural_number(no_iter_max) or no_iter_max <= 0:
        raise Exception("The parameter 'no_iter_max' should be a positive natural number.")

    if not Utils.is_a_number(h) or h < 0:
        raise Exception("The optional parameter 'h' must be a nonnegative number.")

    if not Utils.is_a_number(eps) or eps < 0:
        raise Exception("The optional parameter 'eps' must be a nonnegative number.")
            
            
    if not (old_dist_struct is None):
        try:
            if not (id(old_dist_struct.obj) == id(obj) and id(old_dist_struct.robot) == id(self)):
                Exception("The parameter 'old_dist_struct' is a 'DistStructRobotObj' object, but it " \
                        "must have to be relative to the SAME robot object and SAME external object, and " \
                        "this is not the case.")
        except:
            raise Exception("The parameter 'old_dist_struct' must be a 'DistStructRobotObj' object.")

    if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
        raise Exception("c++ mode is set, but .so file was not loaded!")
    # end error handling

    if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
        if h >0 or eps > 0:
            raise Exception("In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!")
        
        return _compute_dist_python(self, obj, Utils.cvt(q), htm, old_dist_struct, tol, no_iter_max, max_dist)
    else:
        

        old_dsro = _diststructrobotobj_py2cpp(old_dist_struct)    
        new_dsro = self.cpp_robot.compute_dist(obj.cpp_obj, q, htm, old_dsro, tol, no_iter_max, max_dist, h+1e-6, eps+1e-6)

        return _diststructrobotobj_cpp2py(new_dsro, obj, self)
            
        
        

def _compute_dist_python(self, obj, q=None, htm=None, old_dist_struct=None, tol=0.0005, no_iter_max=20,
                  max_dist = np.inf):
    
    n = len(self.links)

    dist_struct = DistStructRobotObj(obj, self)

    jac_dh, mth_dh = self.jac_geo(q, "dh", htm, mode = "python")

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
        for j in range(len(self.links[i].col_objects)):

            est_dist = 0 if math.isinf(max_dist) else Utils.compute_aabbdist(obj, col_object_copy[i][j])

            if est_dist <= max_dist:

                if old_dist_struct is None:
                    p_obj_0 = np.matrix(np.random.uniform(-100, 100, size=(3, 1)))
                else:
                    try:
                        p_obj_0 = Utils.cvt(old_dist_struct.get_item(i, j).point_object)
                    except:
                        p_obj_0 = np.matrix(np.random.uniform(-100, 100, size=(3, 1)))

                p_obj, p_obj_col, d = Utils.compute_dist(obj, col_object_copy[i][j], p_obj_0, tol, no_iter_max)

                jac_obj_col = jac_dh[i][0:3, :] - Utils.S(p_obj_col - mth_dh[i][0:3, 3]) * jac_dh[i][3:6, :]
                jac_dist = ((p_obj_col - p_obj).T * jac_obj_col) / (d+1e-6)

                dist_struct._append(i, j, d, p_obj_col, p_obj, jac_dist)



    return dist_struct
