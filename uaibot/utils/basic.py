import numpy as np
from typing import Tuple
from .types import Matrix, Vector, HTMatrix
from math import cos, sin, acos, sqrt


def cvt(input: Matrix) -> np.matrix:
    if isinstance(input, list) or isinstance(input, tuple):
        if all(isinstance(x, (int, float)) for x in input):
            return np.matrix(input, dtype=np.float64).reshape(-1, 1)
        elif all(isinstance(x, list) for x in input):
            return np.matrix(input, dtype=np.float64)
        else:
            raise ValueError("Entry is not a valid matrix.")

    elif isinstance(input, np.ndarray):
        if input.ndim == 1:
            return np.matrix(input.reshape(-1, 1), dtype=np.float64)
        else:
            if input.shape[1] > 1:
                if input.shape[0] == 1:
                    return np.matrix(input, dtype=np.float64).reshape(-1, 1)
                else:
                    return np.matrix(input, dtype=np.float64)
            else:
                return np.matrix(input, dtype=np.float64)
    else:
        raise TypeError("Entry is not a valid matrix.")


def S(v: Vector) -> np.matrix:
    """
    Returns a 3x3 matrix that implements the cross product for a 3D vector
    as a matricial product, that is, a matrix S(v) such that for any other
    3D column  vector w, S(v)w = cross(v,w).

    Parameters
    ----------
    v : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
        The vector for which the S matrix will be created.

    Returns
    -------
    S : 3x3 numpy matrix
        A matrix that implements the cross product with v.
    """
    vv = cvt(v)
    return np.matrix(
        [[0, -vv[2, 0], vv[1, 0]], [vv[2, 0], 0, -vv[0, 0]], [-vv[1, 0], vv[0, 0], 0]]
    )


def rot(axis: Vector, angle: float) -> HTMatrix:
    """
    Homogeneous transformation matrix that represents the rotation of an
    angle in an axis.

    Parameters
    ----------
    axis : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
        The axis of rotation.

    angle: float
        The angle of rotation, in radians.

    Returns
    -------
    htm : 4x4 numpy matrix
        The homogeneous transformation matrix.
    """
    a = cvt(axis)
    a = a / np.linalg.norm(a)
    K = S(a)
    Q = np.identity(3) + sin(angle) * K + (1 - cos(angle)) * (K * K)
    return np.hstack(
        [np.vstack([Q, np.matrix([0, 0, 0])]), np.matrix([[0], [0], [0], [1]])]
    )


def trn(vector: Vector) -> HTMatrix:
    """
    Homogeneous transformation matrix that represents the displacement
    of a vector

    Parameters
    ----------
    vector : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
        The displacement vector.

    Returns
    -------
    htm : 4x4 numpy matrix
        The homogeneous transformation matrix.
    """
    v = cvt(vector)
    return np.matrix(
        [[1, 0, 0, v[0, 0]], [0, 1, 0, v[1, 0]], [0, 0, 1, v[2, 0]], [0, 0, 0, 1]]
    )


def rotx(angle: float) -> HTMatrix:
    """
    Homogeneous transformation matrix that represents the rotation of an
    angle in the 'x' axis.

    Parameters
    ----------
    angle: float
        The angle of rotation, in radians.

    Returns
    -------
    htm : 4x4 numpy matrix
        The homogeneous transformation matrix.
    """
    return np.matrix(
        [
            [1, 0, 0, 0],
            [0, cos(angle), -sin(angle), 0],
            [0, sin(angle), cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def roty(angle: float) -> HTMatrix:
    """
    Homogeneous transformation matrix that represents the rotation of an
    angle in the 'y' axis.

    Parameters
    ----------
    angle: float
        The angle of rotation, in radians.

    Returns
    -------
    htm : 4x4 numpy matrix
        The homogeneous transformation matrix.
    """
    return np.matrix(
        [
            [cos(angle), 0, sin(angle), 0],
            [0, 1, 0, 0],
            [-sin(angle), 0, cos(angle), 0],
            [0, 0, 0, 1],
        ]
    )


def rotz(angle: float) -> HTMatrix:
    """
    Homogeneous transformation matrix that represents the rotation of an
    angle in the 'z' axis.

    Parameters
    ----------
    angle: float
        The angle of rotation, in radians.

    Returns
    -------
    htm : 4x4 numpy matrix
        The homogeneous transformation matrix.
    """
    return np.matrix(
        [
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def htm_rand(
    trn_min: float = [0.0, 0.0, 0.0],
    trn_max: float = [1.0, 1.0, 1.0],
    rot: float = np.pi / 2,
) -> HTMatrix:
    """
    Returns a random homogeneous transformation matrix.

    Parameters
    ----------
    trn: float
        Maximum parameter for random translation in x, y and z.
        (default: 1)

    rot: float
        Maximum parameter for random rotation in x, y and z.
        (default: 1)

    Returns
    -------
    htm : 4x4 numpy matrix
        A homogeneous transformation matrix.
    """

    x = np.random.uniform(trn_min[0], trn_max[0])
    y = np.random.uniform(trn_min[1], trn_max[1])
    z = np.random.uniform(trn_min[2], trn_max[2])
    ax = np.random.uniform(-rot / 2, rot / 2)
    ay = np.random.uniform(-rot / 2, rot / 2)
    az = np.random.uniform(-rot / 2, rot / 2)

    return trn([x, y, z]) * rotx(ax) * roty(ay) * rotz(az)


def inv_htm(htm: HTMatrix) -> HTMatrix:
    """
    Given a homogeneous transformation matrix, compute its inverse.
    It is faster than using numpy.linalg.inv in the case of HTMs.

    Parameters
    ----------
    htm: 4X4 numpy matrix
        Homogeneous transformation matrix of the rotation.

    Returns
    -------
    inv_htm: 4X4 numpy array
        The inverse of the transformation matrix.
    """
    Q = htm[0:3, 0:3]
    p = htm[0:3, 3]

    inv_htm = np.matrix(np.zeros((4, 4)))
    inv_htm[0:3, 0:3] = Q.T
    inv_htm[0:3, 3] = -Q.T * p
    inv_htm[3, 3] = 1

    return inv_htm


def axis_angle(htm: Matrix) -> Tuple[np.matrix, float]:
    """
    Given an homogeneous transformation matrix representing a rotation,
    return the rotation axis angle.

    Parameters
    ----------
    htm: 4X4 numpy matrix
        Homogeneous transformation matrix of the rotation.

    Returns
    -------
    axis : 3D numpy vector
        The rotation axis.

    angle : float
        The rotation angle, in radians.
    """

    Q = htm[0:3, 0:3]
    trace = Q[0, 0] + Q[1, 1] + Q[2, 2]
    angle = acos((trace - 1) / 2)
    G = Q * Q - 2 * cos(angle) * Q + np.identity(3)
    ok = False
    while not ok:
        v = np.matrix(np.random.uniform(-100, 100, size=(3, 1)))
        w = np.matrix(np.random.uniform(-100, 100, size=(3, 1)))
        r = G * v
        nr = np.linalg.norm(r)
        prod = w.T * r
        if nr > 0.01:
            ortr = w - (r * prod) / (nr * nr)
            axis = S(ortr) * (Q * ortr)
            naxis = np.linalg.norm(axis)
            ok = naxis > 0.01

    axis = axis / naxis
    return axis, angle


def euler_angles(htm: Matrix) -> Tuple[float, float, float]:
    """
    Computer the Euler angles of a rotation matrix.
    Find alpha, beta and gamma such that.

    htm = Utils.rotz(alpha) * Utils.roty(beta) * Utils.rotx(gamma).

    Parameters
    ----------
    htm: 4X4 numpy array or nested list
        Homogeneous transformation matrix of the rotation.

    Returns
    -------
    alpha : float
        Rotation in z, in radians.
    beta : float
        Rotation in y, in radians.
    gamma : float
        Rotation in x, in radians.
    """

    Q = np.matrix(htm[0:3, 0:3])
    sy = sqrt(Q[0, 0] ** 2 + Q[1, 0] ** 2)

    if abs(sy) > 0.001:
        gamma = np.arctan2(Q[2, 1], Q[2, 2])
        beta = np.arctan2(-Q[2, 0], sy)
        alpha = np.arctan2(Q[1, 0], Q[0, 0])
    else:
        gamma = np.arctan2(-Q[1, 2], Q[1, 1])
        beta = np.arctan2(-Q[2, 0], sy)
        alpha = 0

    return alpha, beta, gamma
