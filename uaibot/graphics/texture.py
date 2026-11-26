from uaibot.utils import *
from typing import Optional, Tuple, List

class Texture:
    """
  Texture to be applied in a Mesh Material.
  It is essentially a wrapper of threejs' 'Texture' class.
  See 'https://threejs.org/docs/#api/en/textures/Texture' for more details.

  Parameters
  ----------
  url : string
      The url that contains the texture.
      It must have one of the following formats: 'png', 'bmp', 'jpg', 'jpeg'.

  mapping : string
      Mirrors 'Texture.mapping' in threejs.
      How the image is applied to the object.
      A string  'UVMapping' is the default, where the U,V coordinates are used to apply the map.
      It can be 'UVMapping', 'CubeReflectionMapping', 'CubeRefractionMapping',
      'EquirectangularReflectionMapping', 'EquirectangularRefractionMapping',
      'CubeUVReflectionMapping' or 'CubeUVRefractionMapping'.
      (default: 'UVMapping').

  wrap_s : string
      Mirrors 'Texture.wrapS' in threejs.
      This defines how the texture is wrapped horizontally and corresponds to U in UV mapping.
      The default is 'ClampToEdgeWrapping', where the edge is clamped to the outer edge texels.
      It can be 'RepeatWrapping', 'ClampToEdgeWrapping' or 'MirroredRepeatWrapping'.
      (default: 'ClampToEdgeWrapping').

  wrap_t : string
      Mirrors 'Texture.wrapT' in threejs.
      This defines how the texture is wrapped vertically and corresponds to V in UV mapping.
      The default is 'ClampToEdgeWrapping', where the edge is clamped to the outer edge texels.
      It can be 'RepeatWrapping', 'ClampToEdgeWrapping' or 'MirroredRepeatWrapping'.
      (default: 'ClampToEdgeWrapping').

  mag_filter : string
      Mirrors 'Texture.magFilter' in threejs.
      How the texture is sampled when a texel covers more than one pixel.
      The default is 'LinearFilter', which takes the four closest texels and bilinearly interpolates among them.
      It can be 'NearestFilter' or 'LinearFilter'.
      (default: 'LinearFilter').

  min_filter : string
      Mirrors 'Texture.minFilter' in threejs.
      How the texture is sampled when a texel covers less than one pixel.
      The default is 'LinearMipmapLinearFilter', which uses mipmapping and a trilinear filter.
      It can be 'NearestFilter', 'NearestMipmapNearestFilter', 'NearestMipmapLinearFilter', 'LinearFilter',
      'LinearMipmapNearestFilter' or  'LinearMipmapLinearFilter'.
      (default: 'LinearMipmapLinearFilter').

  offset : 2d array
      Mirrors 'Texture.offset' in threejs.
      How much a single repetition of the texture is offset from the beginning, in each direction U and V.
      Typical range is 0.0 to 1.0.
      (default: [0,0]).

  repeat : 2d array
      Mirrors 'Texture.repeat' in threejs.
      How many times the texture is repeated across the surface, in each direction U and V.
      If repeat is set greater than 1 in either direction, the corresponding Wrap parameter should also be set to
      'RepeatWrapping' or 'MirroredRepeatWrapping' to achieve the desired tiling effect.
      Setting different repeat values for textures is restricted in the same way like .offset.
      (default: [1,1]).

  rotation : float
      How much the texture is rotated around the center point, in radians. Positive values are counter-clockwise.
      (default: 0).

  center : 2d array
      The point around which rotation occurs. A value of (0.5, 0.5) corresponds to the center of the texture.
      Default is [0, 0], the lower left.
      (default: [0, 0]).
  """

    #######################################
    # Attributes
    #######################################

    @property
    def url(self) -> str:
        """The address of the texture."""
        return self._url

    @property
    def mapping(self) -> str:
        """The method for mapping of texture."""
        return self._mapping

    @property
    def wrap_s(self) -> str:
        """The method for wrapping in the U coordinate."""
        return self._wrap_s

    @property
    def wrap_t(self) -> str:
        """The method for wrapping in the V coordinate."""
        return self._wrap_t

    @property
    def mag_filter(self) -> str:
        """The method for the magnification filter."""
        return self._mag_filter

    @property
    def min_filter(self) -> str:
        """The method for the mignification filter."""
        return self._min_filter

    @property
    def offset(self) -> List[float]:
        """The offset in the texture."""
        return self._offset

    @property
    def repeat(self) -> List[float]:
        """The repeat pattern."""
        return self._repeat

    @property
    def rotation(self) -> float:
        """The texture rotation."""
        return self._rotation

    @property
    def center(self) -> List[float]:
        """The shifting for the texture center."""
        return self._center

    #######################################
    # Constructor
    #######################################

    def __init__(self, url: str, mapping: str ="UVMapping", wrap_s: str ="ClampToEdgeWrapping", 
                 wrap_t: str ="ClampToEdgeWrapping", mag_filter: str ="LinearFilter", 
                 min_filter: str ="LinearMipmapLinearFilter", offset: List[float] =[0, 0], 
                 repeat: List[float]=[1, 1], rotation: float =0, center: List[float] =[0, 0])->"Texture":

        
        mapping_list = ['UVMapping', 'CubeReflectionMapping', 'CubeRefractionMapping',
                        'EquirectangularReflectionMapping', 'EquirectangularRefractionMapping',
                        'CubeUVReflectionMapping', 'CubeUVRefractionMapping']
        wrapping_mode_list = ['RepeatWrapping', 'ClampToEdgeWrapping', 'MirroredRepeatWrapping']
        mag_filter_list = ['NearestFilter', 'LinearFilter']
        min_filter_list = ['NearestFilter', 'NearestMipmapNearestFilter', 'NearestMipmapLinearFilter', 'LinearFilter',
                           'LinearMipmapNearestFilter', 'LinearMipmapLinearFilter']



        if not (str(type(mapping)) == "<class 'str'>" and (mapping in mapping_list)):
            raise Exception(
                "The parameter 'mapping' should be one of the following strings: " + str(mapping_list) + ".")

        if not (str(type(wrap_s)) == "<class 'str'>" and (wrap_s in wrapping_mode_list)):
            raise Exception(
                "The parameter 'wrap_s' should be one of the following strings: " + str(wrapping_mode_list) + ".")

        if not (str(type(wrap_t)) == "<class 'str'>" and (wrap_t in wrapping_mode_list)):
            raise Exception(
                "The parameter 'wrap_t' should be one of the following strings: " + str(wrapping_mode_list) + ".")

        if not (str(type(mag_filter)) == "<class 'str'>" and (mag_filter in mag_filter_list)):
            raise Exception(
                "The parameter 'mag_filter' should be one of the following strings: " + str(mag_filter_list) + ".")

        if not (str(type(min_filter)) == "<class 'str'>" and (min_filter in min_filter_list)):
            raise Exception(
                "The parameter 'min_filter_list' should be one of the following strings: " + str(min_filter_list) + ".")

        if not Utils.is_a_vector(offset, 2):
            raise Exception("The parameter 'offset' should be 2D vector.")

        if not Utils.is_a_vector(repeat, 2):
            raise Exception("The parameter 'repeat' should be 2D vector.")

        if not Utils.is_a_number(rotation):
            raise Exception("The parameter 'rotation' should be a float.")

        if not Utils.is_a_vector(center, 2):
            raise Exception("The parameter 'center' should be 2D vector.")

        self._url = url
        self._mapping = mapping
        self._wrap_s = wrap_s
        self._wrap_t = wrap_t
        self._mag_filter = mag_filter
        self._min_filter = min_filter
        self._offset = offset
        self._repeat = repeat
        self._rotation = rotation
        self._center = center

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Texture: '" + self.url + "'"

        return string

    #######################################
    # Methods
    #######################################

    def gen_code(self, name, port):

        image_types = ["png", "bmp", "jpg", "jpeg"]
        error = Utils.is_url_available(Utils.url_modified(self.url,port), image_types)
        if not (error == "ok!"):
            raise Exception("The parameter 'url' " + error)
        
        string = "const texture_" + name + " = new TextureLoader().load('" + Utils.url_modified(self.url,port) + "');\n"
        string += "texture_" + name + ".wrapS = " + str(self.wrap_s) + ";\n"
        string += "texture_" + name + ".wrapT = " + str(self.wrap_t) + ";\n"
        string += "texture_" + name + ".magFilter = " + str(self.mag_filter) + ";\n"
        string += "texture_" + name + ".minFilter = " + str(self.min_filter) + ";\n"
        string += "texture_" + name + ".rotation = " + str(self.rotation) + ";\n"
        string += "texture_" + name + ".offset.set(" + str(self.offset[0]) + "," + str(self.offset[1]) + ");\n"
        string += "texture_" + name + ".repeat.set(" + str(self.repeat[0]) + "," + str(self.repeat[1]) + ");\n"
        string += "texture_" + name + ".center.set(" + str(self.center[0]) + "," + str(self.center[1]) + ");\n\n"

        return string

