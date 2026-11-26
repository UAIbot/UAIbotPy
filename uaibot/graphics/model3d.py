from uaibot.utils import *
from uaibot.graphics.meshmaterial import *
import numpy as np
from uaibot.utils.types import HTMatrix, Matrix, Vector, SimObject



class Model3D:
    """
   A class that represents a 3d model of an object.

   Parameters
   ----------
   url : string
      The url that contains the 3d object.
      It must have one of the following formats: 'obj', 'stl', 'dae'.

   scale : positive float
       The scaling parameter of the object.
       (default: 1).

   htm : 4x4 numpy matrix
       The htm of the 3d models. This is used to tune the 'default' htm for the object in the uaibot simulator.
       This is necessary because the 3d model can have a different 'default' pose than the desired one.
       (default: np.identity(4)).

   mesh_material : 'uaibot.MeshMaterial' object
       The mesh_material to be applied into the 3d model.
       (default: None).
   """

    #######################################
    # Attributes
    #######################################

    @property
    def url(self) -> str:
        """The 3d model url."""
        return self._url

    @property
    def scale(self) -> float:
        """The object scale."""
        return self._scale

    @property
    def htm(self) -> "HTMatrix":
        """Object pose. A 4x4 homogeneous transformation matrix written is scenario coordinates."""
        return self._htm

    @property
    def mesh_material(self) -> "MeshMaterial":
        """The model mesh material."""
        return self._mesh_material

    #######################################
    # Constructor
    #######################################

    def __init__(self, url: str ="", scale: float =1, htm: HTMatrix =np.identity(4), 
                 mesh_material: Optional["MeshMaterial"]=None) -> "Model3D":

        # Error handling
        
        ALLOWED_MESHES_TYPES = ["uaibot.MeshMaterial","uaibot.MTLMeshMaterial","uaibot.GLBMeshMaterial"] 

        if not Utils.is_a_matrix(htm, 4, 4):
            raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

        if not Utils.is_a_number(scale) or scale < 0:
            raise Exception("The parameter 'scale' should be a float.")

        if not (Utils.get_uaibot_type(mesh_material) in ALLOWED_MESHES_TYPES or (mesh_material is None)):
            raise Exception("The parameter 'mesh_material' should be a "+str(ALLOWED_MESHES_TYPES)+" object or 'None'.")

        self._type = url[url.rfind(".")+1:len(url)+1]
        
        if (not self._type == 'obj') and  Utils.get_uaibot_type(mesh_material) == "uaibot.MTLMeshMaterial":
            raise Exception("The parameter 'mesh_material' is a of MTL type, but the object is not of type '.obj'. MTL files are only supported to obj.")

        if (not self._type == 'glb') and  Utils.get_uaibot_type(mesh_material) == "uaibot.GLBMeshMaterial":
            raise Exception("The parameter 'mesh_material' is a GLB type, but the object is not of type '.glb'.")


        # end error handling

        self._url = url
        self._scale = scale
        self._htm = htm
        


        if mesh_material is None:
            self._mesh_material = MeshMaterial()
        else:
            self._mesh_material = mesh_material

    #######################################
    # Std. Print
    #######################################


    def __repr__(self):

        string = "3D object model with url: '" + self.url + "': \n\n"

        return string

    #######################################
    # Methods
    #######################################

   
    def gen_code(self, name, port):
        """Generate code for injection."""
        
        error = Utils.is_url_available(Utils.url_modified(self.url,port), ['obj','stl','dae','glb'])
        if not (error == "ok!"):
            raise Exception("The parameter 'url' " + error)    
        
        
        string = self.mesh_material.gen_code(name, port = port)
        string += "const htm_" + name + " = new Matrix4(); \n"
        string += "htm_" + name + ".set(" \
                  + str(self.htm[0,0]) + "," + str(self.htm[0,1]) + "," + str(self.htm[0,2]) + "," + str(
            self.htm[0,3]) + "," \
                  + str(self.htm[1,0]) + "," + str(self.htm[1,1]) + "," + str(self.htm[1,2]) + "," + str(
            self.htm[1,3]) + "," \
                  + str(self.htm[2,0]) + "," + str(self.htm[2,1]) + "," + str(self.htm[2,2]) + "," + str(
            self.htm[2,3]) + "," \
                  + str(self.htm[3,0]) + "," + str(self.htm[3,1]) + "," + str(self.htm[3,2]) + "," + str(
            self.htm[3,3]) + ");\n"

        string += "const object3d_" + name + "={\n"
        string += "url: '" + Utils.url_modified(self.url,port) + "',\n"
        string += "scale: " + str(self.scale) + ",\n"
        string += "matrix: htm_" + name + ",\n"
        string += "type: '" + self._type + "',\n"
        string += "mesh_material: material_" + name + "};\n\n"

        return string
