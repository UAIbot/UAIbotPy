import numpy as np
import os

from uaibot.utils.types import Metric
from uaibot.utils import (
    cvt,
    is_a_matrix,
    is_a_number,
    is_a_color,
    is_a_groupable_object,
    is_a_list_vector,
    is_a_metric_object,
    is_a_natural_number,
    is_a_name,
    is_a_obj_sim,
    is_a_pd_matrix,
    is_a_simple_object,
    is_a_vector,
    obj_to_cpp,
    compute_dist,
    trn,
    register_cpp_type,
    GeometryKind,
)
from typing import Tuple


class PointCloud(Metric):
    """
    A cloud of points to draw in the simulator.

    There is a fixed set of points ('points' attribute), and animation can be done by choosing, at each time,
    an interval [initial_ind, final_ind] that determines the range of points that will be displayed.

    Parameters
    ----------

    name : string
        The object's name.
        (default: '' (automatic)).

    size : positive float
        The size of each point in the point cloud.

    color : string
        A HTML-compatible color.

    points : a 3xm numpy array or matrix
        A matrix with 3 rows. The first row is the x coordinates, the second row the y coordinates, and the third row
        the z coordinates.

    """

    #######################################
    # Attributes
    #######################################

    @property
    def name(self):
        """The object name."""
        return self._name

    @property
    def size(self):
        """The size of each point in the point cloud."""
        return self._size

    @property
    def color(self):
        """Color of the object, a HTML-compatible string."""
        return self._color

    @property
    def points(self):
        """The points that compose the point cloud."""
        return np.array(self._points)

    @property
    def cpp_obj(self):
        """Used in the c++ interface"""
        return self._cpp_obj

    #######################################
    # Constructor
    #######################################

    @staticmethod
    def try_conversion(vector_list):
        if not isinstance(vector_list, list):
            raise Exception("Error")

        stacked_vectors = []

        for vec in vector_list:

            if not (isinstance(vec, np.ndarray) or isinstance(vec, list)):
                raise Exception("Error")
            if isinstance(vec, list):
                if not (len(vec) == 3):
                    raise Exception("Error")
                else:
                    vec = np.matrix(vec).reshape((3, 1))
            if vec.shape == (3,):
                vec = vec.reshape(3, 1)
            elif vec.shape == (1, 3):
                vec = vec.T
            elif vec.shape == (3, 1):
                pass
            else:
                raise Exception("Error")

            stacked_vectors.append(vec)

        if not stacked_vectors:
            raise Exception("Error")

        return np.matrix(np.hstack(stacked_vectors))

    def __init__(self, name="", points=[], size=0.1, color="blue"):

        # Error handling
        if not is_a_number(size) or size < 0:
            raise Exception("The parameter 'size' should be a positive float")

        if name == "":
            name = "var_pointcloud_id_" + str(id(self))

        if not (is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number."
            )

        if not is_a_color(color):
            raise Exception("The parameter 'color' should be a color")

        if isinstance(points, list) or (not is_a_matrix(points, 3)):
            try:
                self._points = PointCloud.try_conversion(points)
            except:
                raise Exception(
                    "The parameter 'points' should be a matrix with 3 rows or a list in which all the entries are 3D points (numpy or list of values)."
                )
        else:
            self._points = np.matrix(points)

        # end error handling

        self._name = name
        self._size = size
        self._color = color
        self._frames = []
        self._max_time = 0

        if os.environ["CPP_SO_FOUND"] == "1":
            self._cpp_obj = obj_to_cpp(self)
        else:
            self._cpp_obj = []

        self.add_ani_frame(0, 0, np.shape(self._points)[1])

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = (
            "Point cloud '"
            + self._name
            + "' with "
            + str(np.shape(self.points)[1])
            + " points: \n\n"
        )
        string += " Size: " + str(self.size) + "\n"
        string += " Color: " + str(self._color) + "\n"

        return string

    #######################################
    # Methods
    #######################################

    def add_ani_frame(self, time, initial_ind, final_ind):

        if (not is_a_number(time)) or time < 0:
            raise Exception("The parameter 'time' should be a nonnegative float.")

        if (not str(type(initial_ind)) == "<class 'int'>") or initial_ind < 0:
            raise Exception(
                "The parameter 'initial_ind' should be a nonnegative integer."
            )

        if (not str(type(final_ind)) == "<class 'int'>") or final_ind < 0:
            raise Exception(
                "The parameter 'final_ind' should be a nonnegative integer."
            )

        if initial_ind > final_ind:
            raise Exception(
                "The parameter 'initial_ind' should be greater or equal than the parameter 'final_ind'."
            )

        if final_ind > len(self.points[0]):
            raise Exception(
                "The parameter 'final_ind' should be at most "
                + str(len(self.points[0]))
                + "."
            )

        # Error handling

        # end error handling

        self._frames.append([time, initial_ind, final_ind])
        self._max_time = max(self._max_time, time)

    def set_ani_frame(self, time, initial_ind, final_ind):

        if (not is_a_number(time)) or time < 0:
            raise Exception("The parameter 'time' should be a nonnegative float.")

        if (not str(type(initial_ind)) == "<class 'int'>") or initial_ind < 0:
            raise Exception(
                "The parameter 'initial_ind' should be a nonnegative integer."
            )

        if (not str(type(final_ind)) == "<class 'int'>") or final_ind < 0:
            raise Exception(
                "The parameter 'final_ind' should be a nonnegative integer."
            )

        if initial_ind > final_ind:
            raise Exception(
                "The parameter 'initial_ind' should be greater or equal than the parameter 'final_ind'."
            )

        if final_ind > len(self.points[0]):
            raise Exception(
                "The parameter 'final_ind' should be at most "
                + str(len(self.points[0]))
                + "."
            )

        # Error handling

        # end error handling

        self._frames = []
        self.add_ani_frame(0, initial_ind, final_ind)
        self._max_time = 0

    def gen_code(self, port):
        """Generate code for injection."""

        string = "\n"
        string += "//BEGIN DECLARATION OF THE POINT CLOUD '" + self.name + "'\n\n"
        string += (
            "const var_"
            + self.name
            + " = new PointCloud("
            + str(np.around(np.matrix(self.points), 4).tolist())
            + ", "
            + str(self._frames)
            + ", '"
            + self.color
            + "', "
            + str(self.size)
            + ");\n"
        )
        string += "sceneElements.push(var_" + self.name + ");\n"
        string += "//USER INPUT GOES HERE"

        return string

    def copy(self):
        return PointCloud(
            self.name + "_copy", np.matrix(self.points), self.size, self.color
        )

    def aabb(self, mode="auto"):
        """
        Compute an AABB (axis-aligned bounding box), considering the current orientation of the object.

        Parameters
        ----------
        mode : string
            'c++' for the c++ implementation, 'python' for the python implementation
            and 'auto' for automatic ('c++' is available, else 'python')
            (default: 'auto')

        Returns
        -------
         aab: the AABB as a uaibot.Box object
        """

        if (mode == "c++") or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "1"):
            obj_cpp = self._cpp_obj

        if mode == "c++" and os.environ["CPP_SO_FOUND"] == "0":
            raise Exception("c++ mode is set, but .so file was not loaded!")

        from simobjects.box import Box

        if mode == "python" or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "0"):
            row_mins = self.points.min(axis=1)
            row_maxs = self.points.max(axis=1)
            xmin, ymin, zmin = row_mins
            xmax, ymax, zmax = row_maxs
            w = xmax - xmin
            d = ymax - ymin
            h = zmax - zmin
            center = np.matrix(
                [(xmax + xmin) / 2, (ymax + ymin) / 2, (zmax + zmin) / 2]
            ).transpose()
            return Box(
                name="aabb_" + self.name,
                width=w,
                depth=d,
                height=h,
                htm=trn(center),
                opacity=0.5,
            )
        else:
            aabb = obj_cpp.get_aabb()
            return Box(
                name="aabb_" + self.name,
                width=aabb.lx,
                depth=aabb.ly,
                height=aabb.lz,
                htm=trn(aabb.p),
                opacity=0.5,
            )

    # Compute distance to an object
    def compute_dist(
        self, obj, p_init=None, tol=0.001, no_iter_max=20, h=0, eps=0, mode="auto"
    ):
        """
        Compute Euclidean distance or differentiable distance between two objects.

        If h>0 or eps > 0, it computes the Euclidean distance and it uses GJK's algorithm.

        Else, it computes the differentiable distance through Generalized Alternating Projection (GAP).
        See the paper 'A Differentiable Distance Metric for Robotics Through Generalized Alternating Projection'.
        This only works in c++ mode, though.


        Parameters
        ----------
        obj : an object of type 'MetricObject' (see Utils.IS_METRIC)
            The other object for which we want to compute the distance.

        p_init : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array) or None
            Initial point for closest point in this object. If 'None', is set to random.
            (default: None).

        tol : positive float
            Convergence criterion of GAP: it stops when ||a[k+1]-a[k]|| < tol.
            Only valid when h > 0 or eps > 0.
            (default: 0.001m).

        no_iter_max : positive int
            Maximum number of iterations of GAP.
            Only valid when h > 0 or eps > 0.
            (default: 20 iterations).

        h : nonnegative float
            h parameter in the generalized distance function.
            If h=0 and eps=0, it is simply the Euclidean distance.
            (default: 0).

        eps : nonnegative float
            h parameter in the generalized distance function.
            If h=0 and eps=0, it is simply the Euclidean distance.
            (default: 0).

        mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python').
        (default: 'auto').

        Returns
        -------
        point_this : 3 x 1 numpy matrix
            Closest point (Euclidean or differentiable) in this object.

        point_other : 3 x 1 numpy matrix
            Closest point (Euclidean or differentiable) in the other object.

        distance : float
            Euclidean or differentiable distance.

        hist_error: list of floats
            History of convergence error.

        """

        return compute_dist(self, obj, p_init, tol, no_iter_max, h, eps, mode)

    # Compute the projection of a point into an object
    def projection(self, point, h=0, eps=0, mode="auto") -> Tuple[np.matrix, float]:
        """
        The projection of a point in the object, that is, the
        closest point in the object to a point 'point'.

        Parameters
        ----------
        point : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
            The point for which the projection will be computed.

        h : positive float
            Smoothing parameter (only valid in c++ mode)
            (default: 0).

        eps : positive float
            Smoothing parameter (only valid in c++ mode)
            (default: 0).

        Returns
        -------
         proj_point : 3 x 1 numpy matrix
            The projection of the point 'point' in the object.

         d : positive float
            The distance between the object and 'point'.
        """

        if (h > 0 or eps > 0) and (
            (mode == "python")
            or ((mode == "auto" and os.environ["CPP_SO_FOUND"] == "0"))
        ):
            raise Exception(
                "In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!"
            )

        if not is_a_number(h) or h < 0:
            raise Exception("The optional parameter 'h' must be a nonnegative number.")

        if not is_a_number(eps) or eps < 0:
            raise Exception(
                "The optional parameter 'eps' must be a nonnegative number."
            )

        if not is_a_vector(point, 3):
            raise Exception("The parameter 'point' should be a 3D vector.")

        if mode == "c++" and os.environ["CPP_SO_FOUND"] == "0":
            raise Exception("c++ mode is set, but .so file was not loaded!")

        # end error handling
        if mode == "python" or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "0"):
            raise Exception("Projection to point cloud available only for c++ mode!")
        else:
            pr = self._cpp_obj.projection(cvt(point), h, eps)
            return cvt(pr.proj), pr.dist


register_cpp_type(PointCloud, GeometryKind.POINT_CLOUD)
