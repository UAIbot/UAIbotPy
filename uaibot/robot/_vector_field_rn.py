import numpy as np
import os
from uaibot.utils import *

import os
if os.environ['CPP_SO_FOUND']=="1":
    import uaibot_cpp_bind as ub_cpp

_INVHALFPI = 0.63660
_EPS = 1/10000000

def _vector_field_rn(q, curve, alpha, const_vel, is_closed, gamma, mode='auto'):

    n = np.shape(curve[0])[0]

    # Error handling
    if mode not in ['python','c++','auto']:
        raise Exception("The parameter 'mode' should be 'python,'c++', or 'auto'.")

    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not Utils.is_a_number(alpha) or alpha <= 0:
        raise Exception("The parameter 'alpha' should be a positive float.")

    if not Utils.is_a_number(const_vel):
        raise Exception("The parameter 'const_vel' should be a float.")

    if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")

    # end error handling

    if mode=='python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
        return _vector_field_rn_python(Utils.cvt(q), curve, alpha, const_vel, is_closed, gamma)
    else:
        vf_res =  ub_cpp.vectorfield_rn(Utils.cvt(q), curve, alpha, const_vel, is_closed, gamma)
        return Utils.cvt(vf_res.twist), vf_res.dist, vf_res.index



def _vector_field_rn_python(p, curve, alpha, const_vel, is_closed, gamma):


    vec_n, vec_t, min_dist, ind_min = _compute_ntd(curve, p)

    fun_g = _INVHALFPI * atan(alpha * np.sqrt(min_dist))
    fun_h = sqrt(max(1 - fun_g ** 2, 0))
    abs_const_vel = abs(const_vel)
    sgn = const_vel / (abs_const_vel + 0.00001)
    
    if is_closed:
        mult = 1
    else:
        per =  ind_min/len(curve)
        mult = min(gamma * (1.0-per),1.0)
    

    return abs_const_vel * (fun_g * vec_n + mult * sgn * fun_h * vec_t), min_dist, ind_min


def _compute_ntd(curve, p):
    min_dist = float('inf')
    ind_min = -1


    n = np.shape(curve[0])[0]

    pr = Utils.cvt(p)

    for i in range(len(curve)):
        dist_temp = np.linalg.norm(pr - Utils.cvt(curve[i]))
        if dist_temp < min_dist:
            min_dist = dist_temp
            ind_min = i



    vec_n = Utils.cvt(curve[ind_min]) - pr
    vec_n = vec_n / (np.linalg.norm(vec_n) + _EPS)

    if ind_min == len(curve) - 1:
        vec_t = Utils.cvt(curve[1]) - Utils.cvt(curve[ind_min])
    else:
        vec_t = Utils.cvt(curve[ind_min + 1]) - Utils.cvt(curve[ind_min])

    vec_t = vec_t / (np.linalg.norm(vec_t) + _EPS)
    vec_t = np.matrix(vec_t).reshape((n,1))

    return vec_n, vec_t, min_dist, ind_min
