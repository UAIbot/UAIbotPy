from uaibot.utils import *
import numpy as np
import os

# Function used for task function/task Jacobian
def _task_function(self, htm_tg, q=None, htm=None, mode='auto'):

    if q is None:
        q = self.q

    if htm is None:
        htm = self.htm

    # Error handling
    if mode not in ['python','c++','auto']:
        raise Exception("The parameter 'mode' should be 'python,'c++', or 'auto'.")
       
    if not Utils.is_a_matrix(htm_tg, 4, 4):
        raise Exception("The parameter 'htm_tg' should be a 4x4 homogeneous transformation matrix.")

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    n = len(self.links)
    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    # end error handling
    if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
        raise Exception("c++ mode is set, but .so file was not loaded!")
    # end error handling

    if mode == 'python'  or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
        return _task_function_python(self, htm_tg, Utils.cvt(q), htm)
    else:
        task_res = self.cpp_robot.fk_task(Utils.cvt(q), htm, htm_tg)
        return  Utils.cvt(task_res.task), Utils.cvt(task_res.jac_task)

             
            
def _task_function_python(self, htm_tg, q=None, htm=None):


    n = len(self.links)


    p_des = htm_tg[0:3, 3]
    x_des = htm_tg[0:3, 0]
    y_des = htm_tg[0:3, 1]
    z_des = htm_tg[0:3, 2]

    jac_eef, htm_eef = self.jac_geo(q, "eef", htm, mode='python')
    p_eef = htm_eef[0:3, 3]
    x_eef = htm_eef[0:3, 0]
    y_eef = htm_eef[0:3, 1]
    z_eef = htm_eef[0:3, 2]

    r = np.matrix(np.zeros((6,1)))
    r[0:3,0] = p_eef - p_des
    r[3] = max(1 - x_des.T * x_eef, 0)
    r[4] = max(1 - y_des.T * y_eef, 0)
    r[5] = max(1 - z_des.T * z_eef, 0)

    n = len(self.links)
    jac_r = np.matrix(np.zeros((6, n)))
    jac_r[0:3, :] = jac_eef[0:3, :]
    jac_r[3, :] = x_des.T * Utils.S(x_eef) * jac_eef[3:6, :]
    jac_r[4, :] = y_des.T * Utils.S(y_eef) * jac_eef[3:6, :]
    jac_r[5, :] = z_des.T * Utils.S(z_eef) * jac_eef[3:6, :]



    return r, jac_r
