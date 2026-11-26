from .meshmaterial import *
from .glbmeshmaterial import *
from .mtlmeshmaterial import *
from .model3d import *
from .texture import *

from uaibot.utils.types import register_uaibot_type as _register_uaibot_type

# These classes still participate in legacy display-label checks in Model3D
# and MeshMaterial. Register them with the same identity system as scene objects.
for _type in (MeshMaterial, GLBMeshMaterial, MTLMeshMaterial, Model3D, Texture):
    _register_uaibot_type(_type)
