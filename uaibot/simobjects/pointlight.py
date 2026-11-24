import numpy as np
from uaibot.utils import *
from uaibot.utils.types import HTMatrix, Matrix, Vector, MetricObject, Groupable
from typing import Optional, Tuple, List

class PointLight(Groupable):
    """
   A point light.

   Parameters
   ----------

   name : string
       The object's name.
       (default: '' (automatic)).

   color : string
       A HTML-compatible color.
       (default: 'white').

   intensity : positive float
       The light intensity.
       (default: 1).

   htm : 4x4 numpy matrix
      The object's configuration.
      (default: the same as the current HTM).

   max_distance : nonnegative float
       The maximum distance in which the light can act, in meters.
       If set to 0, this distance is infinite.
       (default: 0).

   """

    #######################################
    # Attributes
    #######################################

    @property
    def name(self) -> str:
        """The object name."""
        return self._name

    @property
    def color(self) -> str:
        """The light color."""
        return self._color

    @property
    def intensity(self) -> float:
        """The light intensity"""
        return self._intensity

    @property
    def htm(self) -> "HTMatrix":
        """Object pose. A 4x4 homogeneous transformation matrix written is scenario coordinates."""
        return np.matrix(self._htm)

    @property
    def max_distance(self) -> float:
        """The light maximum distance (meters)."""
        return self._max_distance

    #######################################
    # Constructor
    #######################################

    def __init__(self, htm: "HTMatrix" =np.identity(4), name: str ="", color: str ="white", 
                 intensity: float =1, max_distance: float =0) -> "PointLight":

        # Error handling

        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if name=="":
            name="var_pointlight_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a color.")

        if not Utils.is_a_number(intensity) or intensity < 0:
            raise Exception("The parameter 'intensity' should be a nonnegative float.")

        if not Utils.is_a_number(max_distance) or max_distance < 0:
            raise Exception("The parameter '_max_distance' should be a nonnegative float.")

        # end error handling

        self._name = name
        self._color = color
        self._intensity = intensity
        self._htm = htm
        self._max_distance = max_distance
        self._frames = []
        self._max_time = 0

        self.add_ani_frame(0, self.htm, self.color, self.intensity, self.max_distance)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Point light with name '" + self.name + "': \n\n"
        string += " Color: " + str(self.color) + "\n"
        string += " Position: " + str(self.htm[0:3,3].tolist()) + "\n"
        string += " Intensity: " + str(self.intensity) + "\n"
        string += " Maximum distance: " + str(self.max_distance) + "\n"

        return string

    #######################################
    # Methods
    #######################################

    def add_ani_frame(self, time: float, htm: Optional[HTMatrix]=None, 
                      color: Optional[str] = None, intensity: Optional[float] = None, 
                      max_distance: Optional[float] = None) -> None:
        """
    Add a single configuration to the object's animation queue.

    Parameters
    ----------
    time: positive float
        The timestamp of the animation frame, in seconds.

    htm : 4x4 numpy matrix
        The object's configuration
        (default: the same as the current HTM).

    color : string
       A HTML-compatible color.
       (default: the same as the current color).

    intensity : positive float
       The light intensity.
       (default: the same as the current intensity).

    max_distance : nonnegative float
       The maximum distance in which the light can act, in meters.
       If set to 0, this distance is infinite.
       (default: the same as the current max_distance).

    Returns
    -------
    None
    """
        if color is None:
            color = self.color
        if intensity is None:
            intensity = self.intensity
        if htm is None:
            htm = self._htm
        if max_distance is None:
            max_distance = self.max_distance

        # Error handling
        if not Utils.is_a_number(time) or time < 0:
            raise Exception("The parameter 'time' should be a positive float.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a color.")

        if not Utils.is_a_number(intensity) or intensity < 0:
            raise Exception("The parameter 'intensity' should be a nonnegative float.")

        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(max_distance) or max_distance < 0:
            raise Exception("The parameter '_max_distance' should be a nonnegative float.")
        # end error handling

        f = [time, color, intensity, max_distance, htm[0:3, 3].tolist()]

        self._color = color
        self._intensity = intensity
        self._htm = htm
        self._max_distance = max_distance
        self._frames.append(f)
        self._max_time = max(self._max_time, time)

    # Set config. Restart animation queue
    def set_ani_frame(self, htm: Optional[HTMatrix]=None, 
                      color: Optional[str] = None, intensity: Optional[float] = None, 
                      max_distance: Optional[float] = None) -> None:
        """
    Reset object's animation queue and add a single configuration to the
    object's animation queue.

    Parameters
    ----------

    htm : 4x4 numpy array
        The object's configuration
        (default: the same as the current HTM).

    color : string
       A HTML-compatible color.
       (default: the same as the current color).

    intensity : positive float
       The light intensity.
       (default: the same as the current intensity).

    max_distance : nonnegative float
       The maximum distance in which the light can act, in meters.
       If set to 0, this distance is infinite.
       (default: the same as the current max_distance).


    Returns
    -------
    None
    """

        if color is None:
            color = self.color
        if intensity is None:
            intensity = self.intensity
        if htm is None:
            htm = self._htm
        if max_distance is None:
            max_distance = self.max_distance

        # Error handling
        if not Utils.is_a_number(time) or time < 0:
            raise Exception("The parameter 'time' should be a positive float.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a color.")

        if not Utils.is_a_number(intensity) or intensity < 0:
            raise Exception("The parameter 'intensity' should be a nonnegative float.")

        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(max_distance) or max_distance < 0:
            raise Exception("The parameter '_max_distance' should be a nonnegative float.")
        # end error handling

        self._frames = []
        self.add_ani_frame(0, htm, color, intensity, max_distance)
        self._max_time = 0


    def gen_code(self, port):
        """Generate code for injection."""

        string = "\n"
        string += "//BEGIN DECLARATION OF THE POINT LIGHT '" + self.name + "'\n\n"
        string += "const var_" + self.name + " = new PointLightSim(" + str(self._frames) + ");\n"
        string += "sceneElements.push(var_" + self.name + ");\n"
        string += "//USER INPUT GOES HERE"
        return string
