import numpy as np
from uaibot.utils import *
from uaibot.graphics.texture import *


class GLBMeshMaterial:
    """
    Mesh Material loaded from a GLB file.
    Some GLB files already include Mesh Material, so this is used only as placeholder.

    Parameters
    ----------
    opacity : float between 0 and 1
        Mirrors 'Material.opacity' in threejs.
        Float in the range of 0.0 - 1.0 indicating how transparent the material is.
        A value of 0.0 indicates fully transparent, 1.0 is fully opaque.
        The attribute 'transparent' is set to true automatically if opacity<1.
        (default: 1).


    """

    #######################################
    # Attributes
    #######################################

    @property
    def opacity(self) -> float:
        """The object opacity."""
        return self._opacity

    #######################################
    # Constructor
    #######################################

    def __init__(self, opacity: float =1) -> "GLBMeshMaterial":

        # Error handling

        if not Utils.is_a_number(opacity) or opacity < 0 or opacity > 1:
            raise Exception(
                "The parameter 'opacity' should be a float between 0 (fully transparent) and 1 (fully opaque).")

        self._opacity = opacity


    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "MTL Mesh Material"

        return string

    #######################################
    # Methods
    #######################################

    def gen_code(self, name, port):

        string = "const material_"+ name + " = {"
        
        string+= "type: 'glb',"
        string+= "opacity: "+str(self.opacity)+"};\n\n"
        



        return string
