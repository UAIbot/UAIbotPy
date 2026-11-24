from uaibot.utils import *
import numpy as np

def _check_free_config(self, q=None, htm=None, obstacles=[],
                              check_joint=True, check_auto=True,
                              tol=0.0005, dist_tol=0.005, no_iter_max=20, mode='auto'):
    
    n = len(self.links)

    if q is None:
        q = self.q
    if htm is None:
        htm = self.htm

    # Error handling
    if mode not in ['python','c++','auto']:
        raise Exception("The parameter 'mode' should be 'python,'c++', or 'auto'.")

    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not str(type(obstacles)) == "<class 'list'>":
        raise Exception("The parameter 'list' should be a list of simple objects.")

    for obs in obstacles:
        if not Utils.is_a_metric_object(obs):
            raise Exception("The parameter 'list' should be a list of metric objects : " + str(Utils.IS_METRIC) + ".")

    if not str(type(check_joint)) == "<class 'bool'>":
        raise Exception("The parameter 'check_joint' should be a boolean.")

    if not str(type(check_auto)) == "<class 'bool'>":
        raise Exception("The parameter 'check_auto' should be a boolean.")

    if not Utils.is_a_number(tol) or tol <= 0:
        raise Exception("The parameter 'tol' should be a positive number.")

    if not Utils.is_a_number(dist_tol) or dist_tol <= 0:
        raise Exception("The parameter 'dist_tol' should be a positive number.")

    if not Utils.is_a_natural_number(no_iter_max) or no_iter_max <= 0:
        raise Exception("The parameter 'no_iter_max' should be a positive natural number.")

    if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
        raise Exception("c++ mode is set, but .so file was not loaded!")
    # end error handling

    if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
        return _check_free_config_python(self, Utils.cvt(q), htm, obstacles, check_joint, check_auto, tol, dist_tol, no_iter_max)
    else:
        obstacles_cpp=[]
        for obs in obstacles:
            if Utils.get_uaibot_type(obs) == 'uaibot.CPP_GeometricPrimitives':
                obstacles_cpp.append(obs)
            else:
                obstacles_cpp.append(obs.cpp_obj)
                
        cfcres = self.cpp_robot.check_free_configuration(Utils.cvt(q), htm, obstacles_cpp, check_joint, check_auto, tol, dist_tol, no_iter_max)
        return cfcres.isfree, cfcres.message, cfcres.info


        
def _check_free_config_python(self, q=None, htm=None, obstacles=[],
                              check_joint=True, check_auto=True,
                              tol=0.0005, dist_tol=0.005, no_iter_max=20):
    
    n = len(self.links)

    #_, mth_dh = self.jac_geo(q, "dh", htm)
    mth_dh = self.fkm(q, "dh", htm, mode='python')

    col_object_copy = []

    # Update all collision objects of all links
    for i in range(n):
        col_object_copy.append([])
        for j in range(len(self.links[i].col_objects)):
            temp_copy = self.links[i].col_objects[j][0].copy()
            htmd = self.links[i].col_objects[j][1]
            temp_copy.set_ani_frame(mth_dh[i][:, :] * htmd)
            col_object_copy[i].append(temp_copy)


    #Create the list of all possible auto collision check
    list_auto_col_check = []
    for i in range(n):
        for j in range(i+2,n):
            for isub in range(len(self.links[i].col_objects)):
                for jsub in range(len(self.links[j].col_objects)):
                    list_auto_col_check.append([i,isub,j,jsub])

    #Create the list of all possible collisions
    list_col_check = []
    for i in range(n):
        for isub in range(len(self.links[i].col_objects)):
            for j in range(len(obstacles)):
                list_col_check.append([i,isub,j])

    #Check if there is any violation
    collided = False
    message = "Ok!"
    info = []

    # Check joint limits:
    if check_joint:
        k = 0
        while not collided and k < n:
            collided = q[k] < self.joint_limit[k, 0]
            if collided:
                message = "Joint number " + str(k) + " is below minimum limit."
                info = [0, k]

            k += 1

        k = 0
        while not collided and k < n:
            collided = q[k] > self.joint_limit[k, 1]
            if collided:
                message = "Joint number " + str(k) + " is above maximum limit."
                info = [1, k]

            k += 1

    # Check collision with the obstacles
    k = 0
    while not collided and k < len(list_col_check):

        i = list_col_check[k][0]
        isub = list_col_check[k][1]
        j = list_col_check[k][2]

        if Utils.compute_aabbdist(obstacles[j], col_object_copy[i][isub]) == 0:
            _, _, d, _ = Utils.compute_dist(obstacles[j], col_object_copy[i][isub] \
                                                     , None, tol, no_iter_max, mode='python')
            if d < dist_tol:
                collided = True
                message = "Collision between link " + str(i) + " (col object " + str(isub) + ") and obstacle "+str(j)+"."
                info = [2, i, isub, j]

        k += 1


    # Check auto collision:
    if check_auto:
        k = 0
        while not collided and k < len(list_auto_col_check):

            i = list_auto_col_check[k][0]
            isub = list_auto_col_check[k][1]
            j = list_auto_col_check[k][2]
            jsub = list_auto_col_check[k][3]

            if Utils.compute_aabbdist(col_object_copy[i][isub], col_object_copy[j][jsub]) == 0:

                _, _, d, _ = Utils.compute_dist(col_object_copy[i][isub], col_object_copy[j][jsub] \
                                                         , None, tol, no_iter_max, mode='python')
                
                if d < dist_tol:
                    collided = True
                    message = "Collision between link " + str(i) + " (col object " + str(isub) + ") and link " + str(
                        j) + " " \
                             "(col object " + str(jsub) + ")."
                    info = [3, i, isub, j, jsub]

            k += 1




    return (not collided), message, info
