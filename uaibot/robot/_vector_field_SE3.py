import numpy as np
import os
from uaibot.utils import Utils

if os.environ['CPP_SO_FOUND'] == "1":
    import uaibot_cpp_bind as ub_cpp

_DELTA = 1e-3
_DS = 1e-3

def _vector_field_SE3(state, curve, kt1=1.0, kt2=1.0, kt3=1.0, kn1=1.0, kn2=1.0, curve_derivative=[], delta=_DELTA, ds=_DS, mode='auto'):
    n = 4
    # Error handling
    if mode not in ['python', 'c++', 'auto']:
        raise Exception("The parameter 'mode' should be 'python,'c++', or 'auto'.")

    if not Utils.is_a_matrix(state, n, n):
        raise Exception(f"The parameter 'state' should be a {n}x{n} matrix.")
    if not Utils.is_a_number(kt1):
        raise Exception("The parameter 'kt1' should be a float.")
    if not Utils.is_a_number(kt2):
        raise Exception("The parameter 'kt2' should be a float.")
    if not Utils.is_a_number(kt3):
        raise Exception("The parameter 'kt3' should be a float.")
    if not Utils.is_a_number(kn1):
        raise Exception("The parameter 'kn1' should be a float.")
    if not Utils.is_a_number(kn2):
        raise Exception("The parameter 'kn2' should be a float.")
    if not Utils.is_a_number(delta) or delta <= 0:
        raise Exception("The parameter 'delta' should be a positive float.")
    if not Utils.is_a_number(ds) or ds <= 0:
        raise Exception("The parameter 'delta' should be a positive float.")

    if mode == 'c++' and os.environ['CPP_SO_FOUND'] == '0':
        raise ImportError("c++ mode is set, but .so file was not loaded!")
    if mode == 'python' or (mode == 'auto' and os.environ['CPP_SO_FOUND'] == '0'):
        vf_res = _vector_field_SE3_python(state, curve, kt1, kt2, kt3, kn1, kn2, curve_derivative, delta, ds)
        return None, None, None
    else:
        vf_res = ub_cpp.vectorfield_SE3(state, curve, kt1, kt2, kt3, kn1, kn2, curve_derivative, delta, ds)
        return vf_res.twist, vf_res.dist, vf_res.index

def _vector_field_SE3_python(pstate, curve, kt1=1.0, kt2=1.0, kt3=1.0, kn1=1.0, kn2=1.0, curve_derivative=[], delta=_DELTA, ds=_DS):
    raise NotImplementedError("The python-only version has not been implemented yet.")
    return None

