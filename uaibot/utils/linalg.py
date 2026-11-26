import numpy as np
from typing import Optional
from .types import Matrix, Vector
from .basic import cvt
from ._arrays import real_array
from ._backends import get_backend


def _matrix(value):
    """Preserve 2-D shape; cvt intentionally normalizes vectors to columns."""
    array = real_array(value)
    if array is None or array.ndim not in (1, 2) or array.size == 0:
        raise TypeError("Entry is not a valid real matrix")
    return np.matrix(
        array.reshape(-1, 1) if array.ndim == 1 else array, dtype=np.float64, copy=True
    )


def dp_inv(A: Matrix, eps: float = 0.001) -> np.matrix:
    """
    Compute the damped pseudoinverse of the matrix 'mat'.

    Parameters
    ----------
    A: a matrix ((n,m)-element list/tuple, (n,m)-shaped numpy matrix/numpy array)
        The matrix to compute the damped pseudoinverse.

    eps: positive float
        The damping factor.
        (default: 0.001).

    Returns
    -------
    pinvA: mxn numpy array
        The damped pseudoinverse of 'mat'.
    """
    A_int = _matrix(A)
    n = np.shape(A_int)[1]
    return np.linalg.inv(A_int.T * A_int + eps * np.identity(n)) * A_int.T


def dp_inv_solve(
    A: Matrix, b: Vector, eps: float = 0.001, mode: str = "auto"
) -> np.matrix:
    """
    Solve the problem of minimizing ||A*x-b||^2 + eps*||x||^2
    It is the same as dp_inv(A,eps)*b, but faster

    mode='python' never imports the extension. 'cpp'/'c++' requires it;
    'auto' prefers an importable extension and otherwise falls back to Python.

    Parameters
    ----------
    A: a matrix ((n,m)-element list/tuple, (n,m)-shaped numpy matrix/numpy array)
        A Matrix.

    b: a nD vector (n-element list/tuple, (n,1)/(1,n)/(n,)-shaped numpy matrix/numpy array)
        b Vector.

    eps: positive float
        The damping factor.
        (default: 0.001).

    Returns
    -------
    x: mx1 numpy array
        The solution to the optimization problem.
    """

    backend = get_backend(mode)

    A_cvt = _matrix(A)
    b_cvt = cvt(b)
    n, m = A_cvt.shape

    if backend is None:

        M = np.block([[eps * np.eye(m), -A_cvt.T], [A_cvt, np.eye(n)]])

        rhs = np.concatenate((np.zeros((m, 1)), b_cvt))
        solution = np.linalg.solve(M, rhs)

        return cvt(solution[:m])
    else:
        return cvt(backend.dp_inv_solve(A_cvt, b_cvt, eps))


def solve_qp(
    H: Matrix,
    f: Vector,
    A: Optional[Matrix] = None,
    b: Optional[Vector] = None,
    A_eq: Optional[Matrix] = None,
    b_eq: Optional[Vector] = None,
) -> np.matrix:
    """
    Solve the convex quadratic optimization problem:
    min_u 0.5 * u'Hu + f'u such that A_eq * u = b_eq and A * u >= b.

    Parameters
    ----------
    H : (n,n)-shaped matrix (list/tuple/np.matrix/np.ndarray)
        The H matrix. Must be symmetric positive definite.

    f : (n,)-shaped vector (list/tuple/np.matrix/np.ndarray)
        The f vector.

    A : (m,n)-shaped matrix or 'None', optional
        Inequality constraint matrix (Au >= b). Pass 'None' to omit.

    b : (m,)-shaped vector or 'None', optional
        RHS of inequality constraints. Pass 'None' to omit.

    A_eq : (p,n)-shaped matrix or 'None', optional
        Equality constraint matrix (A_eq * u = b_eq). Pass 'None' to omit.

    b_eq : (p,)-shaped vector or 'None', optional
        RHS of equality constraints. Pass 'None' to omit.

    Returns
    -------
    u : (n,1) numpy matrix
        The solution.
    """

    if (A is None) and (A_eq is None):
        return -np.linalg.inv(_matrix(H)) * (cvt(f))

    H_cvt = _matrix(H)
    H_cvt = 0.5 * (H_cvt + H_cvt.T)  # Ensure symmetry
    f_cvt = np.array(np.asarray(f).reshape((-1,)), dtype=np.float64)

    # Handle empty A and b
    if (A is None) or (b is None):
        A_ineq = np.empty((0, H_cvt.shape[0]))
        b_ineq = np.empty((0,))
    else:
        A_ineq = _matrix(A)
        b_ineq = np.array(np.asarray(b).reshape((-1,)), dtype=np.float64)

    # Handle empty A_eq and b_eq
    if (A_eq is None) or (b_eq is None):
        A_eq_cvt = np.empty((0, H_cvt.shape[0]))
        b_eq_cvt = np.empty((0,))
    else:
        A_eq_cvt = _matrix(A_eq)
        b_eq_cvt = np.array(np.asarray(b_eq).reshape((-1,)), dtype=np.float64)

    # Stack constraints and compute meq
    G = np.vstack([A_eq_cvt, A_ineq])
    h = np.hstack([b_eq_cvt, b_ineq])
    meq = A_eq_cvt.shape[0]

    import quadprog  # Optional dependency; only needed for constrained QPs.

    result = quadprog.solve_qp(H_cvt, -f_cvt, G.T, h, meq)
    return cvt(result[0])


def null_space(A: Matrix) -> np.matrix:
    """
    Compute the null space of the matrix A.

    Parameters:
    ----------
    A : a matrix ((n,m)-element list/tuple, (n,m)-shaped numpy matrix/numpy array)
        The matrix to compute the null space.

    Returns
    -------
    A_null : m x r numpy matrix
        The matrix such that its column spam the null space.
    """
    array = real_array(A)
    if array is None or array.ndim != 2:
        raise TypeError("A must be a real 2-D matrix")
    _, singular, vh = np.linalg.svd(
        np.asarray(array, dtype=np.float64), full_matrices=True
    )
    tolerance = (
        singular[0] * max(array.shape) * np.finfo(np.float64).eps
        if singular.size
        else 0.0
    )
    rank = np.count_nonzero(singular > tolerance)
    return np.matrix(vh[rank:].T, copy=True)
