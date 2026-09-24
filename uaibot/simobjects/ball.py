from uaibot.utils import *
import numpy as np

from uaibot.graphics.meshmaterial import *
from uaibot.simobjects.box import *
from uaibot.simobjects.pointcloud import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from uaibot.simobjects.pointcloud import PointCloud
    from uaibot.simobjects.box import Box
    from uaibot.graphics.meshmaterial import MeshMaterial
    
import os
from uaibot.utils.types import HTMatrix, Matrix, Vector, MetricObject, Simple
from typing import Optional, Tuple, List

class Ball(Simple):
    """
  A ball object.

  Parameters
  ----------
  htm : 4x4 numpy matrix
      The object's configuration.
      (default: the same as the current HTM).

  name : string
      The object's name.
      (default: '' (automatic)).

  radius : positive float
      The object's radius, in meters.
      (default: 1).    

  color : string
      The object's color, a HTML - compatible string.
      (default: "red").

  opacity : float between 0 and 1
      The opacity. 1 = fully opaque, and 0 = transparent.

  mesh_material: 'MeshMaterial' object
      The object mesh material. If set to 'None', the default is used.
      (default: None).
  """

    #######################################
    # Attributes
    #######################################

    @property
    def radius(self) -> float:
        """The ball radius, in meters."""
        return self._radius

    @property
    def name(self) -> str:
        """Name of the object."""
        return self._name

    @property
    def htm(self) -> "HTMatrix":
        """Object pose. A 4x4 homogeneous transformation matrix written is scenario coordinates."""
        return np.matrix(self._htm)

    @property
    def color(self) -> str:
        """Color of the object"""
        return self.mesh_material.color

    @property
    def mesh_material(self) -> "MeshMaterial":
        """Mesh material properties of the object."""
        return self._mesh_material

    @property
    def cpp_obj(self):
        """Used in the c++ interface"""
        self._cpp_obj.htm = self._htm
        return self._cpp_obj

    #######################################
    # Constructor
    #######################################

    def __init__(self, htm: "HTMatrix" =np.identity(4), name: str ="", radius: float =1, color: str ="red", 
                 opacity: float =1, mesh_material: Optional["MeshMaterial"] =None) -> "Ball":

        # Error handling
        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(radius) or radius < 0:
            raise Exception("The parameter 'radius' should be a positive float.")

        if name=="":
            name="var_ball_id_"+str(id(self))

        if not (Utils.is_a_name(name)):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a HTML-compatible color.")

        if not ((mesh_material is None) or (Utils.get_uaibot_type(mesh_material) == "uaibot.MeshMaterial")):
            raise Exception(
                "The parameter 'mesh_material' should be either 'None' or a 'uaibot.MeshMaterial' object.")

        if (not Utils.is_a_number(opacity)) or opacity < 0 or opacity > 1:
            raise Exception("The parameter 'opacity' should be a float between 0 and 1.")
        # end error handling

        self._radius = radius
        self._htm = np.matrix(htm)
        self._name = name
        self._frames = []
        self._max_time = 0

        if mesh_material is None:
            self._mesh_material = MeshMaterial(color=color, opacity=opacity)
        else:
            self._mesh_material = mesh_material

        # Set initial total configuration
        self.set_ani_frame(self._htm)
        
        if os.environ['CPP_SO_FOUND']=="1":
            self._cpp_obj =  Utils.obj_to_cpp(self)

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Ball with name '" + self.name + "': \n\n"
        string += " Radius (m): " + str(self.radius) + "\n"
        string += " Color: " + str(self.color) + "\n"
        string += " HTM: \n" + str(self.htm) + "\n"

        return string

    #######################################
    # Methods
    #######################################

    def add_ani_frame(self, time: float, htm : Optional["HTMatrix"] = None) -> None:
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
        string += "//BEGIN DECLARATION OF THE BALL '" + self.name + "'\n\n"
        string += self.mesh_material.gen_code(self.name, port=port) + "\n"
        string += "const var_" + self.name + " = new Ball(" + str(self.radius) + "," + str(
            self._frames) + ", material_" + self.name + ");\n"
        string += "sceneElements.push(var_" + self.name + ");\n"
        string += "//USER INPUT GOES HERE"

        return string

    def copy(self) -> "Ball":
        """Return a deep copy of the object, without copying the animation frames."""
        return Ball(self.htm, self.name + "_copy", self.radius, self.color)

    def aabb(self, mode: str ='auto') -> "Box":
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
     aabb: the AABB as a uaibot.Box object
    """
    
        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            obj_cpp = self.cpp_obj
            
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")
        
        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            drad = 2 * self.radius
            return Box(name = "aabb_"+self.name, width= drad, depth=drad, height=drad, htm=Utils.trn(self.htm[0:3,-1]),opacity=0.5)
        else:
            aabb = obj_cpp.get_aabb()
            return Box(name = "aabb_"+self.name, width= aabb.lx, depth=aabb.ly, height=aabb.lz, htm=Utils.trn(aabb.p),opacity=0.5)       
            
        

    def to_point_cloud(self, disc: float =0.025, mode: str = 'auto') -> "PointCloud":
        """
    Transform the object into a PointCloud object using the discretization 'delta'.

    Parameters
    ----------
    
    disc: positive float
        Discretization.
        (default: 0.025)

    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto') 
            
    Returns
    -------
     pointcloud: the pointcloud object.
    """

        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            obj_cpp = self.cpp_obj 
            
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")
        
        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            N = round(self.radius / disc)+1
            P = np.matrix(np.zeros((3, 0)))
            for i in range(N):
                phi =  np.pi*(i/N)
                M = round(N*np.sin(phi))+1
                for j in range(M):
                    theta = 2 * np.pi *(j/M)
                    x = self.radius * np.sin(phi) * np.cos(theta)
                    y = self.radius * np.sin(phi) * np.sin(theta)
                    z = self.radius * np.cos(phi)
                    P = np.block([P, np.matrix([x, y,  z]).transpose()])

            for i in range(np.shape(P)[1]):
                P[:,i] = self.htm[0:3,0:3]*P[:,i]+self.htm[0:3,-1]

            return PointCloud(points = P, color = self.color, size = disc)
        else:
            return PointCloud(points = obj_cpp.to_pointcloud(disc).points_gp, color = self.color, size = disc)
      
    # Compute distance to an object
    def compute_dist(self, obj: MetricObject,  p_init: Optional[Vector] = None, 
                     tol: float =0.001, no_iter_max: int =20, h: float =0, 
                     eps: float = 0, mode: str ='auto') -> Tuple[Vector, Vector, float, List]:
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
            
        return Utils.compute_dist(self, obj, p_init, tol, no_iter_max, h, eps, mode)
        
        
    def projection(self, point: Vector, h: float =0, eps: float = 0, mode: str ='auto') -> Tuple[np.matrix, float]:
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
         
    mode : string
        'c++' for the c++ implementation, 'python' for the python implementation
        and 'auto' for automatic ('c++' is available, else 'python')
        (default: 'auto')        
    Returns
    -------
     proj_point : 3 x 1 numpy matrix
        The projection of the point 'point' in the object.

     d : positive float
        The distance between the object and 'point'.
    """


        if (mode == 'c++') or (mode=='auto' and os.environ['CPP_SO_FOUND']=='1'):
            obj_cpp = self.cpp_obj
            
        if ( ( h > 0 or eps > 0) and ((mode == 'python') or ((mode=='auto' and os.environ['CPP_SO_FOUND']=='0')))):
            raise Exception("In Python mode, smoothing parameters 'h' and 'eps' must be set to 0!")
               
        if not Utils.is_a_number(h) or h < 0:
            raise Exception("The optional parameter 'h' must be a nonnegative number.")

        if not Utils.is_a_number(eps) or eps < 0:
            raise Exception("The optional parameter 'eps' must be a nonnegative number.")
        
        if not Utils.is_a_vector(point, 3):
            raise Exception("The parameter 'point' should be a 3D vector.")
        
        if mode=='c++' and os.environ['CPP_SO_FOUND']=='0':
            raise Exception("c++ mode is set, but .so file was not loaded!")

        # end error handling
        
        point_cvt = Utils.cvt(point)

        if mode == 'python' or (mode=='auto' and os.environ['CPP_SO_FOUND']=='0'):
            dd = np.linalg.norm(point_cvt - self._htm[0:3, 3])
            d = max(dd-self.radius,0)

            if d == 0:
                return point_cvt, d
            else:
                cp = self._htm[0:3, 3] + self.radius * (point_cvt - self._htm[0:3, 3]) / dd
                return cp, d
        else:
            pr = obj_cpp.projection(point_cvt, h, eps)
            return Utils.cvt(pr.proj), pr.dist


