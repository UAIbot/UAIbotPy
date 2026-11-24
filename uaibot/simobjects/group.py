import numpy as np
from uaibot.utils import *
from uaibot.utils.types import HTMatrix, Matrix, Vector, MetricObject, GroupableObject, Groupable
from typing import Optional, Tuple, List

class Group(Groupable):
    """
   A group of other 'simobject' objects.

   When the objects are grouped, it is assumed that the scenario frame is the parent frame of all objects.
   This means that when the group is created, it stores the htm of each object at that moment, htm_base_object[i].
   When the htm of the whole group is changed to 'htm', it changes the htm of each member i of the group to
   htm @ htm_base_object[i]. Thus, be sure that, when the group is created, all the objects are in the correct pose.

   To see which type of objects are groupable, see the constant 'Utils.IS_GROUPABLE'.

   Parameters
   ----------
   htm : 4x4 numpy matrix
       The object's configuration.
       (default: the same as the current HTM).

   name : string
       The object's name.
       (default: '' (automatic)).

   list_of_objects : list of objects that are in 'Utils.IS_GROUPABLE'
       List of objects. Each must be an object of the type contained in the constant list 'Utils.IS_GROUPABLE'.

   """

    #######################################
    # Attributes
    #######################################

    @property
    def name(self) -> str:
        """Name of the object."""
        return self._name

    @property
    def list_of_objects(self) -> List["GroupableObject"]:
        """The list of objects contained in the group."""
        return self._list_of_objects

    @property
    def htm(self) -> "HTMatrix":
        """Group pose. A 4x4 homogeneous transformation matrix written is scenario coordinates."""
        return self._htm

    @property
    def htm_base_object(self) -> List["HTMatrix"]:
        """The pose of each object, written in scenario coordinates, when the group was created."""
        return self._htm_base_object

    #######################################
    # Constructor
    #######################################

    def __init__(self, list_of_objects: List["GroupableObject"], name: str="", 
                 htm: "HTMatrix" =np.identity(4)) -> "Group":

        # Error handling

        if name=="":
            name="var_group_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

        if not ((str(type(list_of_objects)) == "<class 'list'>")):
            raise Exception(
                "The parameter 'list_of_objects' should be a list of simple objects (Utils.is_a_simple_object(obj)==True).")

        for obj in list_of_objects:
            if not (Utils.is_a_groupable_object(obj)):
                raise Exception(
                    "The parameter 'list_of_objects' should be a list of the following objects: " + str(
                        Utils.IS_GROUPABLE) + ".")

        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        for i in range(len(list_of_objects)):
            if list_of_objects[i].name == name:
                raise Exception("The name of the group ('" + name + "') is the same name of one of its elements.")

            for j in range(i + 1, len(list_of_objects)):
                if list_of_objects[i].name == list_of_objects[j].name:
                    raise Exception("The are two objects in the list (indexes " + str(i) + " and " + str(
                        j) + ") that have the same name.")

        # end error handling

        self._name = name
        self._list_of_objects = list_of_objects
        self._htm = htm
        self._max_time = 0

        self._htm_base_object = []
        for obj in self.list_of_objects:
            self._htm_base_object.append(obj.htm)

        # Set initial configuration
        self.set_ani_frame(self.htm)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Group '" + self.name + "': \n\n"
        string += "List of objects: "
        for obj in self.list_of_objects:
            string += "'" + obj.name + "', "

        return string

    #######################################
    # Methods
    #######################################

    def add_ani_frame(self, time: float, htm: Optional["HTMatrix"] = None) -> None:
        """
    Add a single configuration to the object's animation queue.

    Parameters
    ----------
    time: positive float
        The timestamp of the animation frame, in seconds.
    htm : 4x4 numpy matrix
        The object's configuration
        (default: the same as the current HTM).

    Returns
    -------
    None
    """

        for i in range(len(self.list_of_objects)):
            self.list_of_objects[i].add_ani_frame(time=time, htm=htm @ self.htm_base_object[i])
            self._max_time = max(self._max_time, self.list_of_objects[i]._max_time)

    def set_ani_frame(self, htm: Optional["HTMatrix"] = None) -> None:
        """
    Reset object's animation queue and add a single configuration to the
    object's animation queue.

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The object's configuration
        (default: the same as the current HTM).

    Returns
    -------
    None
    """

        for i in range(len(self.list_of_objects)):
            self.list_of_objects[i].set_ani_frame(htm=htm @ self.htm_base_object[i])

        self._max_time = 0

    def gen_code(self, port):
        """Generate code for injection."""

        self._max_time = 0
        for i in range(len(self.list_of_objects)):
            self._max_time = max(self._max_time, self.list_of_objects[i]._max_time)

        string = "\n"
        string += "//BEGIN DECLARATION OF THE GROUP '" + self.name + "'\n\n"
        for obj in self.list_of_objects:
            string += obj.gen_code(port=port) + "\n"

        string = string.replace("//USER INPUT GOES HERE", "")
        string += "\n//USER INPUT GOES HERE"

        return string
