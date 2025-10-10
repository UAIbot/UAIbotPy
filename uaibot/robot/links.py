import numpy as np
from utils import *
from graphics.model3d import *


class Link:
    """
  A class that contains the information of the links of the robot.


  Parameters
  ----------

  joint_number : positive int
      The joint number in the kinematic chain (0 is the first joint).

  theta : float
      The 'theta' parameter in the Denavit-Hartenberg convention (rotation in z), in rad.

  d : float
      The 'd' parameter in the Denavit-Hartenberg convention (displacement in z), in meters.

  alpha : float
      The 'alpha' parameter in the Denavit-Hartenberg convention (rotation x), in rad.

  a : float
      The 'a' parameter in the Denavit-Hartenberg convention (displacement in x), in meters.

  joint_type : 0 or 1
      The joint type. "0" is rotative and "1" is prismatic.

  list_model_3d : A list of 'uaibot.Model3D' objects
      The 3d models that compose the links.

  """

    #######################################
    # Attributes
    #######################################

    @property
    def theta(self):
        """The 'theta' parameter of the Denavit-Hartenberg convention (in rad)"""
        return self._theta

    @property
    def d(self):
        """The 'd' parameter of the Denavit-Hartenberg convention (in meters)"""
        return self._d

    @property
    def a(self):
        """The 'a' parameter of the Denavit-Hartenberg convention (in meters)"""
        return self._a

    @property
    def alpha(self):
        """The 'alpha' parameter of the Denavit-Hartenberg convention (in rad)"""
        return self._alpha

    @property
    def joint_number(self):
        """The joint number in the kinematic chain."""
        return self._joint_number

    @property
    def joint_type(self):
        """The joint type (0=revolute, 1=prismatic)."""
        return self._joint_type

    @property
    def col_objects(self):
        """Collection of objects that compose the collision model of the link."""
        return self._col_objects

    @property
    def list_model_3d(self):
        """The list of 3d models of the object."""
        return self._list_model_3d

    @property
    def show_frame(self):
        """If the Denavit-Hartenberg frame of the link is displayed in simulation."""
        return self._show_frame

    @property
    def mass(self):
        """The mass of the link (in kg)."""
        return self._mass

    @property
    def center_of_mass(self):
        """The center of mass of the link (in meters)."""
        return self._center_of_mass

    @property
    def inertia_tensor(self):
        """The inertia tensor of the link (in kg.m^2)."""
        return self._inertia_tensor

    #######################################
    # Constructor
    #######################################

    def __init__(self, joint_number, theta, d, alpha, a, joint_type, list_model_3d, show_frame=False,
                    mass=0.0, center_of_mass=np.zeros(3), inertia_tensor=np.zeros((3, 3))):

        # Error handling
        if str(type(joint_number)) != "<class 'int'>" or joint_number < 0:
            raise Exception("The 'joint_number' parameter should be a nonnegative integer.")

        if not Utils.is_a_number(theta):
            raise Exception("The 'theta' parameter should be a float.")

        if not Utils.is_a_number(d):
            raise Exception("The 'd' parameter should be a float.")

        if not Utils.is_a_number(alpha):
            raise Exception("The 'alpha' parameter should be a float.")

        if not Utils.is_a_number(a):
            raise Exception("The 'a' parameter should be a float.")

        if joint_type != 0 and joint_type != 1:
            raise Exception("The 'joint_type' parameter should be either '0' (rotative) or '1' (prismatic).")

        if not (str(type(list_model_3d)) == "<class 'list'>"):
            raise Exception("The parameter 'list_model_3d' should be a list of 'uaibot.Model3D' objects.")
        else:
            for i in range(len(list_model_3d)):
                if not (Utils.get_uaibot_type(list_model_3d[i]) == "uaibot.Model3D"):
                    raise Exception(
                        "The parameter 'list_model_3d' should be a list of 'uaibot.Model3D' objects.")

        if not str(type(show_frame)) == "<class 'bool'>":
            raise Exception("The parameter 'show_frame' should be a boolean.")

        # Code
        self._joint_number = joint_number
        self._theta = theta
        self._d = d
        self._alpha = alpha
        self._a = a
        self._joint_type = joint_type
        self._col_objects = []
        self._list_model_3d = list_model_3d
        self._show_frame = show_frame
        self._mass = mass
        self._center_of_mass = np.array(center_of_mass).ravel()
        self._inertia_tensor = np.array(inertia_tensor)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Link " + str(self._joint_number) + "' "

        if self._joint_type == 0:
            string += "with rotative joint:\n\n"
            string += " θ (rad) : [variable] \n"
            string += " d (m)   : " + str(self._d) + " \n"
        if self._joint_type == 1:
            string += "with prismatic joint:\n\n"
            string += " θ (rad) : " + str(self._theta) + " \n"
            string += " d (m)   : [variable] \n"

        string += " α (rad) : " + str(self._alpha) + " \n"
        string += " a (m)   : " + str(self._a) + " \n"

        return string

    #######################################
    # Methods
    #######################################

    def attach_col_object(self, obj, htm):
        """
    Attach an object (ball, box or cylinder) into the link as a collision
    object.
    
    Parameters
    ----------
    obj: object
        Object to be added.

    htm : 4x4 numpy array
        The transformation between the link's HTM and the object's HTM

    """

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not (Utils.is_a_simple_object(obj)):
            raise Exception("The parameter 'obj' must be one of the following types: " + str(Utils.IS_SIMPLE) + ".")

        self._col_objects.append([obj, htm])

    def gen_code(self, name, port):
        """Generate code for injection."""

        name_link = "link_" + str(self.joint_number) + "_" + name

        string = "const object3d_" + name_link + "_list = [];\n\n"

        for i in range(len(self._list_model_3d)):
            string += self.list_model_3d[i].gen_code(name_link + "_obj_" + str(i), port=port)
            string += "object3d_" + name_link + "_list.push(object3d_" + name_link + "_obj_" + str(i) + ");\n"

        string += "\n"
        string += "const " + name_link + " = {\n"
        string += "theta: " + str(self.theta) + ", \n"
        string += "d: " + str(self.d) + ", \n"
        string += "a: " + str(self.a) + ", \n"
        string += "alpha: " + str(self.alpha) + ", \n"
        string += "jointType: " + str(self.joint_type) + ", \n"
        string += "showFrame: " + ("true" if self.show_frame else "false") + ",\n"
        string += "model3d: object3d_" + name_link + "_list}; \n \n"

        return string
