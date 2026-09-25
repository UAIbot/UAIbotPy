import numpy as np
from typing import List, Callable
from math import floor
from .basic import cvt
from .types import Vector
from .typecheck import is_a_vector, is_a_number, is_a_list_vector

# from math import cos, sin, acos, sqrt


def interpolate(
    points: List[Vector], is_closed: bool = False
) -> Callable[[List[float]], List[np.matrix]]:
    """
    Create a function handle that generates an one-time differentiable interpolated data from 'points'.

    The simplest case in when 'points' is a list with m elements. In this case, it will output a function f.
    When this function is evaluated at a scalar t, it will coincide with points[i] when t = i/m, that is,
    f(i/m) = points[i]. This function is once differentiable and periodic with period 1, so f(t+k)=f(t) for
    an integer k.

    The function can also use a n x m numpy array or lists as 'points'. In this case, f(t) is a n x 1
    numpy vector.

    Finally, t can be a list of k elements instead of just a scalar. In this case, f(t) is a list of n x 1 numpy
    matrices, in which the k-th element is the same as computing f(t[k]).


    Parameters
    ----------
    points: a list of nD vectors (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        Points to be interpolated.

    is_closed: bool
        If the curve is closed or not (that is, f(0)=f(1)).
        (default: False)

    Returns
    -------
    f: function handle
        The function handle that implements the interpolation.
    """

    if not is_a_list_vector(points):
        raise Exception(
            "The parameter 'points' should be a list of vectors with the same dimension."
        )

    def aux_interpolate_single(arg_points):
        def int_aux_simp(arg_t, arg_n, arg_c):
            tn = arg_n * (arg_t % 1 if is_closed else min(max(arg_t, 0), 1))
            ti = min(floor(tn), arg_n - 1)
            tloc = tn - ti
            coef = arg_c[4 * ti : 4 * ti + 4]
            return coef[0] + coef[1] * tloc + coef[2] * tloc**2 + coef[3] * tloc**3

        def int_aux(arg_t, arg_n, arg_c):
            return (
                [int_aux_simp(tt, arg_n, arg_c) for tt in arg_t]
                if isinstance(arg_t, list)
                else int_aux_simp(arg_t, arg_n, arg_c)
            )

        n = len(arg_points)
        num_segments = n if is_closed else n - 1
        A_rows = []
        b_vals = []

        # Interpolation constraints (value at start and end of each segment)
        for i in range(num_segments):
            row_start = np.zeros(4 * num_segments)
            row_end = np.zeros(4 * num_segments)
            row_start[4 * i : 4 * i + 4] = [1, 0, 0, 0]
            row_end[4 * i : 4 * i + 4] = [1, 1, 1, 1]
            A_rows.append(row_start)
            b_vals.append(arg_points[i])
            A_rows.append(row_end)
            b_vals.append(arg_points[(i + 1) % n if is_closed else i + 1])

        # Derivative continuity at junctions (including wraparound if closed)
        for i in range(num_segments):
            i_next = (i + 1) % num_segments
            if not is_closed and i == num_segments - 1:
                break  # skip final derivative constraint for open curve
            row = np.zeros(4 * num_segments)
            row[4 * i + 1 : 4 * i + 4] = [1, 2, 3]
            row[4 * i_next + 1 : 4 * i_next + 4] = [-1, 0, 0]
            A_rows.append(row)
            b_vals.append(0)

        # Assemble matrices
        A = np.vstack(A_rows)
        b = np.array(b_vals).reshape(-1, 1)

        # Regularizer: minimize integral of second derivative squared
        H = np.zeros((4 * num_segments, 4 * num_segments))
        for i in range(num_segments):
            H_i = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 6], [0, 0, 6, 12]])
            H[4 * i : 4 * i + 4, 4 * i : 4 * i + 4] = H_i

        # Solve constrained minimization via KKT system
        m = A.shape[0]
        KKT = np.block([[H, A.T], [A, np.zeros((m, m))]])
        rhs = np.vstack([np.zeros((4 * num_segments, 1)), b])
        sol = np.linalg.solve(KKT, rhs)

        c = sol[: 4 * num_segments].flatten().tolist()
        return lambda s: int_aux(s, num_segments, c)

    def aux_interpolate_multiple(arg_points, t):

        if not (is_a_vector(t) or is_a_number(t)):
            raise Exception(
                "The parameter of the interpolation function must be either a number or a list of numbers."
            )

        y = np.zeros((0, len(t) if is_a_vector(t) else 1))
        for i in range(np.shape(cvt(arg_points[0]))[0]):
            fun = aux_interpolate_single([cvt(p)[i, 0] for p in arg_points])
            fun_out = fun(t)
            fun_out = np.array(fun_out).reshape(
                (1, len(fun_out) if is_a_vector(fun_out) else 1)
            )
            y = np.block([[y], [fun_out]])

        list_of_points = [np.matrix(y[:, i].reshape(-1, 1)) for i in range(y.shape[1])]
        if isinstance(t, list):
            return list_of_points
        else:
            return list_of_points[0]

    return lambda t: aux_interpolate_multiple(points, t)
