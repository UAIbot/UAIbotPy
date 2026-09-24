from uaibot.utils import *
import numpy as np
from uaibot.utils.types import HTMatrix, Matrix, Vector, MetricObject, Groupable
from typing import Optional, Tuple, List

from uaibot.graphics.model3d  import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from uaibot.graphics.model3d import Model3D


class RigidObject(Groupable):
    """
   A rigid object.

   Parameters
   ----------

   name : string
       The object's name.
       (default: '' (automatic)).

   list_model_3d : list of 'Model3D' objects
       The 3d model that compose the object.

   htm : 4x4 numpy matrix
       The object's configuration.
       (default: the same as the current HTM).

   """

    #######################################
    # Attributes
    #######################################

    @property
    def name(self) -> str:
        """Name of the object."""
        return self._name

    @property
    def htm(self) -> "HTMatrix":
        """Object pose. A 4x4 homogeneous transformation matrix written is scenario coordinates."""
        return np.matrix(self._htm)

    @property
    def list_model_3d(self) -> List["Model3D"]:
        """The list of 3d models of the object."""
        return self._list_model_3d

    #######################################
    # Constructor
    #######################################

    def __init__(self, list_model_3d: List["Model3D"], name: str ="", 
                 htm: "HTMatrix"=np.identity(4)) -> "RigidObject":

        # Error handling

        if name=="":
            name="var_rigidobject_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. "
                "It should not begin with a number.")

        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not (str(type(list_model_3d)) == "<class 'list'>"):
            raise Exception("The parameter 'list_model_3d' should be a list of 'uaibot.Model3D' objects.")
        else:
            for i in range(len(list_model_3d)):
                if not (Utils.get_uaibot_type(list_model_3d[i]) == "uaibot.Model3D"):
                    raise Exception(
                        "The parameter 'list_model_3d' should be a list of 'uaibot.Model3D' objects.")

        # end error handling

        self._htm = np.matrix(htm)
        self._name = name
        self._list_model_3d = list_model_3d
        self._max_time = 0

        self.set_ani_frame(self._htm)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Rigid Object with name '" + self.name + "': \n\n"
        string += " HTM: \n" + str(self.htm) + "\n"

        return string

    #######################################
    # Methods
    #######################################

    def add_ani_frame(self, time: float, htm: Optional["HTMatrix"]=None) -> None:
        """
    Add a single configuration to the object's animation queue.

    Parameters
    ----------
    time: positive float
        The timestamp of the animation frame, in seconds.
    htm : 4x4 numpy array
        The object's configuration
        (default: the same as the current HTM).

    Returns
    -------
    None
    """
        if htm is None:
            htm = self._htm

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(time) or time < 0:
            raise Exception("The parameter 'time' should be a positive float.")
        # end error handling

        f = [time, np.around(htm[0,0],4).item(), np.around(htm[0,1],4).item(), np.around(htm[0,2],4).item(), np.around(htm[0,3],4).item(),
             np.around(htm[1,0],4).item(), np.around(htm[1,1],4).item(), np.around(htm[1,2],4).item(), np.around(htm[1,3],4).item(),
             np.around(htm[2,0],4).item(), np.around(htm[2,1],4).item(), np.around(htm[2,2],4).item(), np.around(htm[2,3],4).item(),
             0, 0, 0, 1]

        self._htm = htm
        self._frames.append(f)
        self._max_time = max(self._max_time, time)

    # Set config. Restart animation queue
    def set_ani_frame(self, htm: Optional["HTMatrix"]=None) -> None:
        """
    Reset object's animation queue and add a single configuration to the
    object's animation queue.

    Parameters
    ----------
    htm : 4x4 numpy array
        The object's configuration
        (default: the same as the current HTM).

    Returns
    -------
    None
    """

        if htm is None:
            htm = self._htm

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        # end error handling

        self._frames = []
        self.add_ani_frame(0, htm)
        self._max_time = 0

    def gen_code(self, port):
        """Generate code for injection."""

        string = "\n"
        string += "//BEGIN DECLARATION OF THE RIGID OBJECT '" + self.name + "'\n\n"

        string += "var list_object3d_" + self.name + " = [];\n"
        for i in range(len(self.list_model_3d)):
            string += self.list_model_3d[i].gen_code(self.name + "_" + str(i),port=port) + ";\n"
            string += "list_object3d_" + self.name + ".push(object3d_" + self.name + "_" + str(i) + ");\n"

        string += "const var_" + self.name + " = new RigidObject(" + str(
            self._frames) + ", list_object3d_" + self.name + ");\n"
        string += "sceneElements.push(var_" + self.name + ");\n"
        string += "//USER INPUT GOES HERE"

        return string
