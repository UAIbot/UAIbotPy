from uaibot.simobjects.ball import *
from uaibot.simobjects.group import *
from uaibot.simobjects.cylinder import *
from uaibot.graphics.meshmaterial import *
from uaibot.utils import *
import numpy as np
from typing import Optional, Tuple, List
from uaibot.utils.types import Groupable

GroupableObject: TypeAlias = Union["Ball", "Box", "Cylinder", "ConvexPolytope", "Frame",
                    "RigidObject", "Group", "Robot", "PointLight"]


class Frame(Groupable):
    """
  A frame object.

  Parameters
  ----------
  htm : 4x4 numpy matrix
      The object's configuration.
      (default: the same as the current HTM).

  name : string
      The object's name.
      (default: '' (automatic)).

  size : positive float
      The axis sizes, in meters.
      (default: 0.3).

  axis_color : list of 3 HTML-compatible strings
      A list of 3 HTML-compatible strings, one for each axis.
      (default: ['red', 'lime', 'blue']).

  axis_names : list of 3 string
      The axis names.
      (default: ['x', 'y', 'z']).
  """

    #######################################
    # Attributes
    #######################################

    @property
    def size(self) -> float:
        """The axis size, in meters."""
        return self._size

    @property
    def name(self) -> str:
        """Name of the object."""
        return self._name

    @property
    def htm(self) -> "HTMatrix":
        """Object pose. A 4x4 homogeneous transformation matrix written is scenario coordinates."""
        return np.matrix(self._ball.htm)

    @property
    def axis_color(self) -> List[str]:
        """The axis colors. It is a list of 3 HTML-compatible colors."""
        return self._axis_color

    @property
    def axis_name(self) -> List[str]:
        """The axis names. It is a list of 3 strings."""
        return self._axis_name


    #######################################
    # Constructor
    #######################################

    def __init__(self, htm: "HTMatrix" =np.identity(4), name: str ="", size: float =0.3, 
                 axis_color: List[str] =['red', 'lime', 'blue'],
                 axis_names: List[str] =['x', 'y', 'z']) -> "Frame":

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(size) or size <= 0:
            raise Exception("The parameter 'size' should be a positive float.")

        if name=="":
            name="var_frame_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

        if not (str(type(axis_color)) == "<class 'list'>") or (not (len(axis_color) == 3)):
            raise Exception("The parameter 'list' should be a a list of 3 HTML-compatible color strings.")

        for color in axis_color:
            if not Utils.is_a_color(color):
                raise Exception("The parameter 'list' should be a a list of 3 HTML-compatible color strings.")

        # end error handling

        self._name = name
        self._size = size
        self._axis_names = axis_names
        self._axis_color = axis_color
        self._ball = Ball(name="dummy_ball_" + name, htm=htm, radius=0.0001, mesh_material=MeshMaterial(opacity=0))
        self._max_time = 0
        
        cyl_x = Cylinder(radius=0.004, height=size, color = axis_color[0],htm=Utils.roty(np.pi/2)*Utils.trn([0,0,size/2]))
        cyl_y = Cylinder(radius=0.004, height=size, color = axis_color[1],htm=Utils.rotx(-np.pi/2)*Utils.trn([0,0,size/2]))
        cyl_z = Cylinder(radius=0.004, height=size, color = axis_color[2],htm=Utils.trn([0,0,size/2]))
        self._axis_group = Group([cyl_x, cyl_y, cyl_z])

        # Set initial total configuration
        self.set_ani_frame(np.matrix(htm))

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Axes with name '" + self.name + "': \n\n"
        string += " HTM: \n" + str(self.htm) + "\n"

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
    htm : 4x4 numpy array
        The object's configuration.
        (default: the same as the current HTM).

    Returns
    -------
    None
    """

        self._ball.add_ani_frame(time, htm)
        self._axis_group.add_ani_frame(time, htm)
        self._max_time = self._ball._max_time

    # Set config. Restart animation queue
    def set_ani_frame(self, htm: Optional["HTMatrix"] = None) -> None:
        """
    Reset object's animation queue and add a single configuration to the
    object's animation queue.

    Parameters
    ----------
    htm : 4x4 numpy matrix
        The object's configuration.
        (default: the same as the current HTM).

    Returns
    -------
    None
    """

        self._ball.set_ani_frame(htm)
        self._axis_group.set_ani_frame(htm)
        self._max_time = 0

    def gen_code(self, port):
        """Generate code for injection."""
        
        return self._axis_group.gen_code(port=port)

