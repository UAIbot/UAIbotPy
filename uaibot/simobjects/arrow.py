import numpy as np
from utils import *
from uaibot.utils.types import HTMatrix, Matrix, Vector, MetricObject
from typing import Optional, Tuple, List

class Arrow:
    """
   An arrow object (vector).

   Parameters
   ----------

   name : string
       The object's name.
       (default: '' (automatic)).

   color : string
       The object's color, a HTML - compatible string.
       (default: "red").

   thickness : float
       The arrow's thickness.
       (default: 1).

   origin : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
       The origin of the arrow.
       (default: [0,0,0]).

   vector : a 3D vector (3-element list/tuple, (3,1)/(1,3)/(3,)-shaped numpy matrix/numpy array)
       The vector, starting from the origin, of the arrow.
       (default: [0,0,0]).
                     
   """

    #######################################
    # Attributes
    #######################################

    @property
    def name(self) -> str:
        """Name of the object."""
        return self._name

    @property
    def color(self) -> str:
        """Color of the object"""
        return self._color

    @property
    def thickness(self) -> float:
        """The thickness of the vetor."""
        return self._thickness

    @property
    def origin(self) -> Vector:
        """The origin point of the vector, in scenario coordinates and in meters."""
        return self._origin

    @property
    def vector(self) -> Vector:
        """The vector, in scenario coordinates and in meters."""
        return self._vector

    #######################################
    # Constructor
    #######################################

    def __init__(self, name: str ="", color: str ="black", thickness: float =1, 
                 origin: Vector =[0, 0, 0], vector: Vector =[0.5, 0.5, 0.5]) -> "Arrow":

        # Error handling

        if name=="":
            name="var_vector_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. "
                "It should not begin with a number.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a HTML-compatible color.")

        if not Utils.is_a_number(thickness) or thickness <= 0:
            raise Exception("The parameter 'thickness' should be positive float.")

        if not Utils.is_a_vector(origin, 3):
            raise Exception("The parameter 'origin' should be a 3d list or numpy array.")

        if not Utils.is_a_vector(vector, 3):
            raise Exception("The parameter 'vector' should be a 3d list or numpy array.")

        # end error handling

        self._name = name
        self._color = color
        self._thickness = thickness
        self._origin = np.matrix(origin).reshape((3,1))
        self._vector = np.matrix(vector).reshape((3,1))
        self._frames = []
        self._max_time = 0

        # Set initial total configuration
        self.set_ani_frame(self.origin, self.vector)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Arrow '" + self.name + "': \n\n"
        string += "Origin: "+str(self.origin)+" \n"
        string += "Vector: " + str(self.vector) + " \n"

        return string

    #######################################
    # Methods
    #######################################

    def add_ani_frame(self, time: float, origin: Optional[Vector] = None, vector: Optional[Vector] = None) -> None:

        if origin is None:
            origin = self.origin

        if vector is None:
            vector = self.vector

        # Error handling
        if not Utils.is_a_vector(origin, 3):
            raise Exception("The parameter 'origin' should be a 3d list or numpy array.")

        if not Utils.is_a_vector(vector, 3):
            raise Exception("The parameter 'vector' should be a 3d list or numpy array.")

        if not Utils.is_a_number(time) or time < 0:
            raise Exception("The parameter 'time' should be a positive float.")
        # end error handling

        length_vector = np.linalg.norm(np.matrix(vector)).item()

        #if length_vector < 0.0001:
        #    raise Exception("'origin' and 'vector' are too close to each other.")

        self._origin = np.around(np.matrix(origin).reshape((3,1)),4).tolist()
        self._vector = np.matrix(vector).reshape((3,1)).tolist()
        dir = np.around((np.matrix(vector).reshape((3,1)) / (0.0001+length_vector)),4).tolist()

        self._frames.append([time, self.origin, dir, length_vector])
        self._max_time = max(self._max_time, time)

    # Set config. Restart animation queue
    def set_ani_frame(self, start_point: Optional[Vector] = None, end_point : Optional[Vector] = None) -> None:

        self._frames = []
        self.add_ani_frame(0, start_point, end_point)
        self._max_time = 0

    def gen_code(self, port):
        """Generate code for injection."""

        length_vector = np.linalg.norm(np.matrix(self.vector))
        dir = (np.matrix(self.vector) / (length_vector+0.00001)).tolist()

        string = "\n"
        string += "//BEGIN DECLARATION OF THE VECTOR '" + self.name + "'\n\n"
        string += "const var_" + self.name + " = new Vector(" + str(self._frames) + ", '" + str(
            self.color) + "'," + str(self.thickness) + ", " + str(self.origin) + ", " + str(dir) + ", " + str(
            length_vector) + ");\n"
        string += "sceneElements.push(var_" + self.name + ");\n"
        string += "//USER INPUT GOES HERE"

        return string
