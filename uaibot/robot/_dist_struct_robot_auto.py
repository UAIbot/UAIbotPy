import numpy as np
from uaibot.utils.types import HTMatrix, Matrix, Vector, MetricObject
from typing import Optional, Tuple, List

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .robot import Robot

__all__ = ["DistStructRobotAuto", "DistStructLinkLink"]
    
class DistStructLinkLink:

    @property
    def link_number_1(self) -> int:
        """The index of the first link in which the collision point is."""
        return self._link_number_1

    @property
    def link_number_2(self) -> int:
        """The index of the second link in which the collision point is."""
        return self._link_number_2

    @property
    def link_col_obj_number_1(self) -> int:
        """The index of the collision object of the first link in which the collision point is."""
        return self._link_col_obj_number_1

    @property
    def link_col_obj_number_2(self) -> int:
        """The index of the collision object of the second link in which the collision point is."""
        return self._link_col_obj_number_2

    @property
    def distance(self) -> float:
        """The distance (in meters) between the two links."""
        return self._distance

    @property
    def point_link_1(self) -> np.matrix:
        """The closest point (witness) in the first link. Written in scenario coordinates and in meters."""
        return np.matrix(self._point_link_1)

    @property
    def point_link_2(self) -> np.matrix:
        """The closest point (witness) in the second link. Written in scenario coordinates and in meters."""
        return np.matrix(self._point_link_2)

    @property
    def jac_distance(self) -> np.matrix:
        """The Jacobian of the distance in the robot's configuration space."""
        return np.matrix(self._jac_distance)

    #######################################
    # Constructor
    #######################################

    def __init__(self, link_number_1: int, link_col_obj_number_1: int, link_number_2: int, 
                 link_col_obj_number_2: int, distance: float, point_link_1: np.matrix, 
                 point_link_2: np.matrix, jac_distance: np.matrix) -> "DistStructLinkLink":

        self._link_number_1 = link_number_1
        self._link_number_2 = link_number_2
        self._link_col_obj_number_1 = link_col_obj_number_1
        self._link_col_obj_number_2 = link_col_obj_number_2
        self._distance = distance
        self._point_link_1 = point_link_1
        self._point_link_2 = point_link_2
        self._jac_distance = jac_distance

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string  = "Distance info: \n"
        string += " First Link number: "+str(self.link_number_1)+"\n"
        string += " Collision object of the first link number: "+str(self.link_col_obj_number_1 )+"\n"
        string += " Second Link number: "+str(self.link_number_2)+"\n"
        string += " Collision object of the second link number: "+str(self.link_col_obj_number_2 )+"\n"
        string += " Distance: " + str(self.distance) + " m\n"
        string += " Point in the first link: " + str(self.point_link_1.tolist()) + " m\n"
        string += " Point in the second link: " + str(self.point_link_2.tolist()) + " m\n"
        string += " Jacobian distance: " + str(self._jac_distance.tolist()) + "\n"
        return string


class DistStructRobotAuto:

    #######################################
    # Attributes
    #######################################

    @property
    def robot(self) -> "Robot":
        """Return the associated robot."""
        return self._robot

    @property
    def jac_dist_mat(self) -> np.matrix:
        """
		Return the matrix in which each row we have the distance Jacobian (gradient) for each robot link.
		"""
        return np.matrix(self._jac_dist_mat)

    @property
    def dist_vect(self) -> np.matrix:
        """
		Return the column vector in which each row we have the distance for each robot link.
		"""
        return np.matrix(self._dist_vect).reshape((self.no_items, 1))

    @property
    def no_items(self) -> int:
        """Return the number of items."""
        return self._no_items

    def __getitem__(self, key):
        return self._list_info[key]

    #######################################
    # Constructor
    #######################################

    def __init__(self, robot: "Robot") -> "DistStructRobotAuto":

        self._robot = robot
        self._robot_name = robot.name
        self._no_items = 0

        n = len(robot.links)
        self._list_info = []
        self._jac_dist_mat = np.matrix(np.zeros((0, n)))
        self._dist_vect = []

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        return "Distance struct between robot '" + self.robot.name + "' and itself (auto collision) '" \
               " with " + str(self.no_items) + " items"

    #######################################
    # Methods
    #######################################

    def _append(self, link_number_1, link_col_obj_number_1, link_number_2, 
                link_col_obj_number_2, distance, point_link_1, point_link_2, jac_distance) :

        self._list_info.append(
            DistStructLinkLink(link_number_1, link_col_obj_number_1, link_number_2, link_col_obj_number_2,
                               distance, point_link_1, point_link_2, jac_distance))

        self._jac_dist_mat = np.vstack((self._jac_dist_mat, jac_distance))
        self._dist_vect.append(distance)
        self._no_items += 1


    def get_item(self, link_number_1: int, link_col_obj_number_1: int, 
                 link_number_2: int, link_col_obj_number_2: int) -> "DistStructLinkLink":
        for dist_info in self._list_info:
            if link_number_1 == dist_info.link_number_1 and link_col_obj_number_1 == dist_info.link_col_obj_number_1 \
            and link_number_2 == dist_info.link_number_2 and link_col_obj_number_2 == dist_info.link_col_obj_number_2:
                return dist_info

        raise Exception("Item not found!")

    def get_closest_item(self) -> "DistStructLinkLink":
        minimum_distance = float('inf')
        index_minimum = -1
        for i in range(self._no_items):
            if self[i].distance < minimum_distance:
                minimum_distance = self[i].distance
                index_minimum = i

        return self[index_minimum]

