import numpy as np
import string as st
from colour import Color
from .basic import cvt
from .types import Simple, Metric, ObjSim, Groupable
from .typecheck import is_a_vector, is_a_number, is_a_list_vector

    # UAIBOT_NAME_TYPES = ['uaibot.', 'cylinder.', 'box.', 'ball.', 'convexpolytope.', 'robot.', 'simulation.', 'meshmaterial.', 'mtlmeshmaterial.',
    #                          'glbmeshmaterial.', 'texture.', 'pointlight.', 'frame.', 'model3d.', 'links.', 'pointcloud.', 'arrow.', 'rigidobject.',
    #                          '.group', '.htmldiv', 'CPP_GeometricPrimitives', 'CPP_DistStructRobotObj','CPP_AABB']
    #
    # IS_SIMPLE = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope']
    #
    # IS_METRIC = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.PointCloud', 'uaibot.ConvexPolytope']
    #
    # IS_GROUPABLE = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope', 'uaibot.Frame',
    #                 'uaibot.RigidObject', 'uaibot.Group', 'uaibot.Robot', 'uaibot.PointLight']
    #
    # IS_OBJ_SIM = ['uaibot.Ball', 'uaibot.Box', 'uaibot.Cylinder', 'uaibot.ConvexPolytope', 'uaibot.Robot',
    #               'uaibot.PointLight', 'uaibot.Frame', 'uaibot.PointCloud', 'uaibot.Arrow',
    #               'uaibot.RigidObject', 'uaibot.Group', 'uaibot.HTMLDiv']
    #

def is_a_number(obj):
    """
    Check if the argument is a float or int number

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """

    return str(type(obj)) in [
        "<class 'int'>",
        "<class 'float'>",
        "<class 'numpy.float64'>",
    ]


def is_a_natural_number(obj):
    """
    Check if the argument is a natural number (integer and >=0)

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """

    return str(type(obj)) == "<class 'int'>" and obj >= 0


def is_a_matrix(obj, n=None, m=None):
    """
    Check if the argument is an n x m matrix of integers or floats.

    Parameters
    ----------
    obj: object
        Object to be verified.

    n: positive int
        Number of rows
        (default: it does not matter).

    m: positive int
        Number of columns
        (default: it does not matter).

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """

    # Check if obj is a numpy array or matrix
    if isinstance(obj, (np.ndarray, np.matrix)):
        # Check shape if dimensions are provided

        shp = np.shape(obj)
        rows = shp[0]
        cols = shp[1] if len(shp) > 1 else 1

        if ((n is not None) and (rows != n)) or ((m is not None) and (cols != m)):
            return False

        # Ensure all elements are integers or floats
        return np.issubdtype(obj.dtype, np.floating) or np.issubdtype(
            obj.dtype, np.integer
        )

    # Check if obj is a list (and therefore could represent a matrix)
    if isinstance(obj, list):
        try:
            return is_a_matrix(np.matrix(obj), n, m)
        except:
            return False


def is_a_vector(obj, n=None):
    """
    Check if the argument is a vector of floats.
    A vector is very flexible. For example, a 3D vector (is_a_vector(x,3)) can be:
    x1 = [0,1,2]
    x2 = [[0],[1],[2]]
    x3 = (0,1,2)
    x4 = np.matrix([0,1,2])
    x5 = np.matrix([[0],[1],[2]])
    x6 = np.array([0,1,2])
    x7 = np.array([[0],[1],[2]])

    Parameters
    ----------
    obj: object
        Object to be verified.

    n: positive int
        Check if the vector is n-dimensional. If 'None', it does not matter.
        (default: 'None').

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """
    try:
        p = cvt(obj)
        if n is None:
            return np.shape(p)[1] == 1
        else:
            return (np.shape(p)[0] == n) and (np.shape(p)[1] == 1)
    except:
        return False


def is_a_list_vector(obj, n=None):
    """
    Check if the argument is a list of vector of floats with the same dimension.

    Parameters
    ----------
    obj: object
        Object to be verified.

    n: positive int
        Check if all the vectors are n-dimensional. If 'None', it does not matter,
        as long as they are equal.
        (default: 'None').

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """

    if n is None:
        try:
            n_common = np.shape(cvt(obj[0]))[0]
        except:
            return False
    else:
        n_common = n

    for p in obj:
        if not is_a_vector(p, n_common):
            return False

    return True


def is_a_pd_matrix(obj, n=None):
    """
    Check if the argument is a symmetric nxn positive (semi)-definite matrix.

    Parameters
    ----------
    obj: object
        Object to be verified.

    n: positive int
        Dimension of the square matrix
        (default: it does not matter).

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """
    value = is_a_matrix(obj, n, n)

    if value:
        value = np.allclose(obj, np.transpose(obj), rtol=1e-05, atol=1e-08)

    if value:
        try:
            np.linalg.cholesky(obj)
        except:
            value = False

    return value


def is_a_color(obj):
    """
    Check if the argument is a HTML-compatible string that represents a color.

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """

    try:
        obj = obj.replace(" ", "")
        Color(obj)
        return True
    except:
        return False


def is_a_name(string):
    """
    Check if the argument is a valid name for uaibot objects.
    Only characters [a-z], [A-z], [0-9] and '_' are allowed.
    However, variables should not begin with numbers.

    Parameters
    ----------
    string: string
        Name to be verified.

    Returns
    -------
    is_name: boolean
        If the name is a valid name.
    """

    if str(type(string)) == "<class 'str'>":
        if string == "":
            return True
        else:
            allowed1 = set(st.ascii_lowercase + st.ascii_uppercase + st.digits + "_")
            allowed2 = set(st.ascii_lowercase + st.ascii_uppercase + "_")

            return (set(string) <= allowed1) and (set(string[0]) <= allowed2)
    else:
        return False


def get_uaibot_type(obj):
    """
    Return the UAIBot type of the object.
    Return the empty string if it is not an UAIBot object.

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    obj_type: string
        UAIBot type.
    """
    type_str = str(type(obj))

    ind = -1
    k = 0
    while ind == -1 and k < len(Utils.UAIBOT_NAME_TYPES):
        ind = type_str.find(Utils.UAIBOT_NAME_TYPES[k])
        k += 1

    if ind == -1:
        return ""
    else:
        ind1 = type_str.rfind(".")
        ind2 = type_str.rfind(">")
        return "uaibot." + type_str[ind1 + 1 : ind2 - 1]


def is_a_simple_object(obj):
    """
    Check if the argument is a simple object.
    Check the constant 'Utils.IS_SIMPLE' for a list of simple objects.

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """
    return isinstance(obj, Simple)


def is_a_metric_object(obj):
    """
    Check if the argument is a metric object (if distance can be computed).
    Check the constant 'Utils.IS_METRIC' for a list of metric objects.

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """
    return isinstance(obj, Metric)


def is_a_groupable_object(obj):
    """
    Check if the argument is a groupable object.
    Check the constant 'Utils.IS_GROUPABLE' for a list of groupable objects.

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """
    return isinstance(obj, Groupable)


def is_a_obj_sim(obj):
    """
    Check if the argument is an object that can be put into the simulator.
    Check the constant 'Utils.IS_OBJ_SIM' for a list of objects that can be put in the simulator.

    Parameters
    ----------
    obj: object
        Object to be verified.

    Returns
    -------
    is_type: boolean
        If the object is of the type.
    """
    return isinstance(obj, ObjSim)
