import numpy as np
from uaibot.utils import *
from uaibot.graphics.texture import *
from uaibot.utils import *
from .texture import *

class MeshMaterial:
    """
    A class that contains a Mesh Material for applying into objects.
    It is essentially a wrapper of threejs' 'MeshPhysicalMaterial' class.
    See https://threejs.org/docs/#api/en/materials/MeshPhysicalMaterial for
    more details.

    Parameters
    ----------
    opacity : float between 0 and 1
        Mirrors 'Material.opacity' in threejs.
        Float in the range of 0.0 - 1.0 indicating how transparent the material is.
        A value of 0.0 indicates fully transparent, 1.0 is fully opaque.
        The attribute 'transparent' is set to true automatically if opacity<1.
        (default: 1).

    shadow_side : string
        Mirrors 'Material.shadowSide' in threejs.
        Defines which side of faces cast shadows. When set, can be
        ''FrontSide', 'BackSide', or 'DoubleSide' or 'null'.
        (default: 'null').

    side : string
        Mirrors 'Material.side' in threejs.
        Defines which side of faces will be rendered - front, back or both.
        Default is 'FrontSide'. Other options are 'BackSide' and 'DoubleSide'.
        (default: 'FrontSide').

    alpha_map : string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.alphaMap' in threejs.
        The alpha map is a grayscale texture that controls the opacity across the surface
        (black: fully transparent; white: fully opaque).
        If it is a string, a default texture is created from it.
        If it is the empty string, it is ignored.
        (default: empty string).

    ao_map : string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.aoMap' in threejs.
        The red channel of this texture is used as the ambient occlusion map.
        The aoMap requires a second set of UVs.
        If it is a string, a default texture is created from it.
        If it is the empty string, it is ignored.
        (default: empty string).

    ao_map_intensity : float between 0 and 1
        Mirrors 'MeshStandardMaterial.aoMapIntensity' in threejs.
        Intensity of the ambient occlusion effect. Default is 1. Zero is no occlusion effect.
        (default: 1).

    ao_map_intensity : float between 0 and 1
        Mirrors 'MeshStandardMaterial.aoMapIntensity' in threejs.
        Intensity of the ambient occlusion effect. Default is 1. Zero is no occlusion effect.
        (default: 1).

    bump_map : string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.bumpMap' in threejs.
        The texture to create a bump map. The black and white values map to the perceived depth in relation to the lights.
        Bump doesn't actually affect the geometry of the object, only the lighting. If a normal map is defined this will
        be ignored.
        If it is a string, a default texture is created from it.
        If it is the empty string, it is ignored.
        (default: empty string).

    bump_scale : float between 0 and 1
        Mirrors 'MeshStandardMaterial.bumpScale' in threejs.
        How much the bump map affects the material.
        (default: 1).

    color : HTML-compatible color string
        Mirrors 'MeshStandardMaterial.color' in threejs.
        Color of the material.
        (default: "white").

    emissive : HTML-compatible color string
        Mirrors 'MeshStandardMaterial.emissive' in threejs.
        Emissive (light) color of the material, essentially a solid color unaffected by other lighting.
        (default: "black").

    emissive_map : string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.emissiveMap' in threejs.
        Set emisssive (glow) map. Default is null. The emissive map color is modulated by the emissive color and the
        emissive intensity. If you have an emissive map, be sure to set the emissive color to something other
        than black.
        (default: empty string).

    emissive_intensity : positive float
        Mirrors 'MeshStandardMaterial.emissiveIntensity' in threejs.
        Intensity of the emissive light. Modulates the emissive color.
        (default: 1).

    env_map : string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.envMap' in threejs.
        The environment map. To ensure a physically correct rendering, you should only add environment maps which were
        preprocessed by 'PMREMGenerator'.
        (default: empty string).

    env_map_intensity : positive float
        Mirrors 'MeshStandardMaterial.envMapIntensity' in threejs.
        Scales the effect of the environment map by multiplying its color.
        (default: 1).

    flat_shading : boolean
        Mirrors 'MeshStandardMaterial.flatShading' in threejs.
        Define whether the material is rendered with flat shading.
        (default: False).

    light_map : string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.lightMap' in threejs.
        The light map. The lightMap requires a second set of UVs.
        (default: empty string).

    light_map_intensity : positive float
        Mirrors 'MeshStandardMaterial.lightMapIntensity' in threejs.
        Intensity of the baked light.
        (default: 1).

    texture_map: string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.map' in threejs.
        The color map. The texture map color is modulated by the diffuse .color.
        (Obs: the name of this parameter in 'Python' is not 'map' because it is
        a reserved word).
        (default: empty string).

    metalness : float between 0 and 1
        Mirrors 'MeshStandardMaterial.metalness' in threejs.
        How much the material is like a metal. Non-metallic materials such as wood or stone use 0.0, metallic use 1.0,
        with nothing (usually) in between. A value between 0.0 and 1.0 could be used for a rusty metal
        look. If metalnessMap is also provided, both values are multiplied.
        (default: 0).

    metalness_map: string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.metalnessMap' in threejs.
        The blue channel of this texture is used to alter the metalness of the material.
        (default: empty string).

    normal_map: string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.normalMap' in threejs.
        The texture to create a normal map. The RGB values affect the surface normal for each pixel fragment and change
        the way the color is lit. Normal maps do not change the actual shape of the surface, only the lighting. In case
        the material has a normal map authored using the left handed convention, the y component of normalScale should
        be negated to compensate for the different handedness.
        (default: empty string).

    normal_scale: 2D vector
        Mirrors 'MeshStandardMaterial.normalScale' in threejs.
        How much the normal map affects the material. Typical ranges are 0-1.
        (default: [1,1]).

    refraction_ratio : float between 0 and 1
        Mirrors 'MeshStandardMaterial.refractionRatio' in threejs.
        The index of refraction (IOR) of air (approximately 1) divided by the index of refraction of the material. It is
        used with environment mapping modes 'CubeRefractionMapping' and 'EquirectangularRefractionMapping'.
        The refraction ratio should not exceed 1.
        (default: 0.98).

    roughness : float between 0 and 1
        Mirrors 'MeshStandardMaterial.roughness' in threejs.
        How rough the material appears. 0.0 means a smooth mirror reflection, 1.0 means fully diffuse.
        If roughnessMap is also provided, both values are multiplied.
        (default: 1).

    normal_map: string containing an url or a 'Texture' object
        Mirrors 'MeshStandardMaterial.roughnessMap' in threejs.
        The green channel of this texture is used to alter the roughness of the material.
        (default: empty string).

    clearcoat : float between 0 and 1
        Mirrors 'MeshPhysicalMaterial.clearcoat' in threejs.
        Represents the intensity of the clear coat layer, from 0.0 to 1.0. Use clear coat related properties to enable
        multilayer materials that have a thin translucent layer over the base layer.
        (default: 0).

    clearcoat_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.clearcoatMap' in threejs.
        The red channel of this texture is multiplied against .clearcoat, for per-pixel control over a coating's
        intensity.
        (default: empty string).

    clearcoat_normal_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.clearcoatNormalMap' in threejs.
        Can be used to enable independent normals for the clear coat layer.
        (default: empty string).

    clearcoat_normal_scale: 2D vector
        Mirrors 'MeshPhysicalMaterial.clearcoatNormalScale' in threejs.
        How much .clearcoatNormalMap affects the clear coat layer, from [0,0] to [1,1].
        (default: [1,1]).

    clearcoat_roughness : float between 0 and 1
        Mirrors 'MeshPhysicalMaterial.clearcoat_roughness' in threejs.
        Roughness of the clear coat layer, from 0.0 to 1.0.
        (default: 0).

    clearcoat_roughness_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.clearcoatRoughnessMap' in threejs.
        The green channel of this texture is multiplied against .clearcoatRoughness, for per-pixel control over a
        coating's roughness.
        (default: empty string).

    ior : float between 1 and 2.333
        Mirrors 'MeshPhysicalMaterial.ior' in threejs.
        Index-of-refraction for non-metallic materials, from 1.0 to 2.333.
        (default: 1.5).

    reflectivity : float between 0 and 1
        Mirrors 'MeshPhysicalMaterial.reflectivity' in threejs.
        Degree of reflectivity, from 0.0 to 1.0. Default is 0.5, which corresponds to an index-of-refraction of 1.5.
        This models the reflectivity of non-metallic materials. It has no effect when metalness is 1.0
        (default: 0.5).

    sheen : float between 0 and 1
        Mirrors 'MeshPhysicalMaterial.sheen' in threejs.
        The intensity of the sheen layer, from 0.0 to 1.0.
        (default: 0).

    sheen_roughness : float between 0 and 1
        Mirrors 'MeshPhysicalMaterial.sheenRoughness' in threejs.
        Roughness of the sheen layer, from 0.0 to 1.0.
        (default: 1).

    sheen_roughness_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.sheenRoughnessMap' in threejs.
        The alpha channel of this texture is multiplied against .sheenRoughness, for per-pixel control over
        sheen roughness.
        (default: empty string).

    sheen_color : HTML-compatible color string
        Mirrors 'MeshPhysicalMaterial.sheenColor' in threejs.
        The sheen tint.
        (default: "white").

    sheen_color_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.sheenColorMap' in threejs.
        The RGB channels of this texture are multiplied against .sheenColor, for per-pixel control over sheen tint.
        (default: empty string).

    specular_intensity : float between 0 and 1
        Mirrors 'MeshPhysicalMaterial.specularIntensity' in threejs.
        A float that scales the amount of specular reflection for non-metals only.
        When set to zero, the model is effectively Lambertian. From 0.0 to 1.0.
        (default: 0).

    specular_intensity_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.specularIntensityMap' in threejs.
        The alpha channel of this texture is multiplied against .specularIntensity, for per-pixel control over
        specular intensity
        (default: empty string).

    specular_color : HTML-compatible color string
        Mirrors 'MeshPhysicalMaterial.specularColor' in threejs.
        A Color that tints the specular reflection at normal incidence for non-metals only.
        (default: "white").

    specular_color_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.specularColorMap' in threejs.
        The RGB channels of this texture are multiplied against .specularColor, for per-pixel control over specular
        color.
        (default: empty string).

    transmission : float between 0 and 1
        Mirrors 'MeshPhysicalMaterial.transmission' in threejs.
        Degree of transmission (or optical transparency), from 0.0 to 1.0.
        Thin, transparent or semitransparent, plastic or glass materials remain largely reflective even if they are
        fully transmissive. The transmission property can be used to model these materials.
        When transmission is non-zero, opacity should be set to 1.
        (default: 0).

    transmission_map: string containing an url or a 'Texture' object
        Mirrors 'MeshPhysicalMaterial.transmissionMap' in threejs.
        The red channel of this texture is multiplied against .transmission, for per-pixel control over optical
        transparency.
        (default: empty string).
    """

    #######################################
    # Attributes
    #######################################

    @property
    def opacity(self) -> float:
        """The object opacity."""
        return self._opacity

    @property
    def shadow_side(self) -> str:
        """Which side of faces casts shadows."""
        return self._shadow_side

    @property
    def side(self) -> str:
        """Which side of faces will be rendered."""
        return self._side

    @property
    def transparent(self) -> bool:
        """If the object is transparent."""
        return self._transparent

    @property
    def alpha_map(self):
        """The alpha map texture"""
        return self._alpha_map

    @property
    def ao_map(self):
        """The texture for the red channel for the ambient occlusion (ao) map."""
        return self._ao_map

    @property
    def ao_map_intensity(self):
        """The ambient occlusion intensity."""
        return self._ao_map_intensity

    @property
    def bump_map(self):
        """The bump texture map."""
        return self._bump_map

    @property
    def bump_map_scale(self):
        """Bump map scale."""
        return self._bump_map_scale

    @property
    def color(self):
        """The object color."""
        return self._color

    @property
    def emissive(self):
        """The emissivity of the material."""
        return self._emissive

    @property
    def emissive_map(self):
        """The emissivity texture map."""
        return self._emissive_map

    @property
    def emissive_intensity(self):
        """The emissivity intensity."""
        return self._emissive_intensity

    @property
    def env_map(self):
        """Environmental texture map."""
        return self._env_map

    @property
    def env_map_intensity(self):
        """The intensivity of the environmental map."""
        return self._env_map_intensity

    @property
    def flat_shading(self):
        """If the object is flat shaded."""
        return self._flat_shading

    @property
    def light_map(self):
        """The texture map of the light."""
        return self._light_map

    @property
    def light_map_itensity(self):
        """The light map intensity."""
        return self._light_map_itensity

    @property
    def map(self):
        """The main texture map."""
        return self._map

    @property
    def metalness(self):
        """The metalness of the object."""
        return self._metalness

    @property
    def metalness_map(self):
        """The metalness texture map."""
        return self._metalness_map

    @property
    def normal_map(self):
        """The normal map texture."""
        return self._normal_map

    @property
    def normal_scale(self):
        """The normal sale."""
        return self._normal_scale

    @property
    def refraction_ratio(self):
        """The refraction ratio."""
        return self._refraction_ratio

    @property
    def roughness(self):
        """The roughness intensity."""
        return self._roughness

    @property
    def roughness_map(self):
        """The roughness texture map."""
        return self._roughness_map

    @property
    def clearcoat(self):
        """Represents the intensity of the clear coat layer."""
        return self._clearcoat

    @property
    def clearcoat_map(self):
        """Clear coat texture map."""
        return self._clearcoat_map

    @property
    def clearcoat_normal_map(self):
        """Clear coat normal texture map."""
        return self._clearcoat_normal_map

    @property
    def clearcoat_normal_scale(self):
        """Clear coat normal scale."""
        return self._clearcoat_normal_scale

    @property
    def clearcoat_roughness(self):
        """The clear coat roughness."""
        return self._clearcoat_roughness

    @property
    def clearcoat_roughness_map(self):
        """The clear coat roughness texture map."""
        return self._clearcoat_roughness_map

    @property
    def ior(self):
        """Index of refraction of the material"""
        return self._ior

    @property
    def reflectivity(self):
        """The reflectivity of the material."""
        return self._reflectivity

    @property
    def sheen(self):
        """The intensity of the sheen layer."""
        return self._sheen

    @property
    def sheen_roughness(self):
        """The sheen layer roughness."""
        return self._sheen_roughness

    @property
    def sheen_roughness_map(self):
        """The sheen layer roughness texture map."""
        return self._sheen_roughness_map

    @property
    def sheen_color(self):
        """The sheen color."""
        return self._sheen_color

    @property
    def sheen_color_map(self):
        """The sheen color texture map."""
        return self._sheen_color_map

    @property
    def specular_intensity(self):
        """The specular intensity."""
        return self._specular_intensity

    @property
    def specular_intensity_map(self):
        """The specular intensity texture map."""
        return self._specular_intensity_map

    @property
    def specular_color(self):
        """The specular color."""
        return self._specular_color

    @property
    def specular_color_map(self):
        """The specular color texture map."""
        return self._specular_color_map

    @property
    def transmission(self):
        """The material transmission."""
        return self._transmission

    @property
    def transmission_map(self):
        """The material transmission texture map."""
        return self._transmission_map

    #######################################
    # Constructor
    #######################################

    def __init__(self, opacity=1, shadow_side="null", side="null", alpha_map="", ao_map="", \
                 ao_map_intensity=1, bump_map="", bump_map_scale=1, color="white", emissive="black", emissive_map="", \
                 emissive_intensity=1, env_map="", env_map_intensity=1, flat_shading=False, light_map="", \
                 light_map_itensity=1, texture_map="", metalness=0, metalness_map="", normal_map="", \
                 normal_scale=[1, 1], refraction_ratio=0.98, roughness=1, roughness_map="", clearcoat=0, \
                 clearcoat_map="", clearcoat_normal_map="", clearcoat_normal_scale=[1, 1], clearcoat_roughness=0, \
                 clearcoat_roughness_map="", ior=1.5, reflectivity=0.5, sheen=0, sheen_roughness=1,
                 sheen_roughness_map="", \
                 sheen_color="white", sheen_color_map="", specular_intensity=0, specular_intensity_map="",
                 specular_color="white", \
                 specular_color_map="", transmission=0, transmission_map=""):

        # Error handling

        if not Utils.is_a_number(opacity) or opacity < 0 or opacity > 1:
            raise Exception(
                "The parameter 'opacity' should be a float between 0 (fully transparent) and 1 (fully opaque).")

        if not (str(type(shadow_side)) == "<class 'str'>") or not ( \
                        (shadow_side == "FrontSide") or \
                        (shadow_side == "BackSide") or \
                        (shadow_side == "DoubleSide") or \
                        (shadow_side == "null")):
            raise Exception(
                "The parameter 'shadow_side' should be either \"FrontSide\", \"BackSide\", \"DoubleSide\" or \"null\".")

        if not (str(type(side)) == "<class 'str'>") or not ( \
                        (side == "FrontSide") or \
                        (side == "BackSide") or \
                        (side == "DoubleSide") or \
                        (side == "null")):
            raise Exception(
                "The parameter 'side' should be either \"FrontSide\", \"BackSide\" or \"DoubleSide\" or \"null\".")

        image_types = ["png", "bmp", "jpg", "jpeg"]

        if not (alpha_map == "" or Utils.get_uaibot_type(alpha_map) == "uaibot.Texture"):
            alpha_map = Texture(alpha_map)

        if not (ao_map == "" or Utils.get_uaibot_type(ao_map) == "uaibot.Texture"):
            ao_map = Texture(ao_map)

        if not Utils.is_a_number(ao_map_intensity) or ao_map_intensity < 0 or ao_map_intensity > 1:
            raise Exception(
                "The parameter 'ao_map_intensity' should be a float between 0 (no occlusion) and 1 (occlusion).")

        if not (bump_map == "" or Utils.get_uaibot_type(bump_map) == "uaibot.Texture"):
            bump_map = Texture(bump_map)

        if not Utils.is_a_number(bump_map_scale) or bump_map_scale < 0:
            raise Exception("The parameter 'bump_map_scale' should be a float greater or equal than 0.")

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a HTML-style color.")

        if not Utils.is_a_color(emissive):
            raise Exception("The parameter 'emissive' should be a HTML-style color.")

        if not (emissive_map == "" or Utils.get_uaibot_type(emissive_map) == "uaibot.Texture"):
            emissive_map = Texture(emissive_map)

        if not Utils.is_a_number(emissive_intensity) or emissive_intensity < 0:
            raise Exception("The parameter 'emissive_intensity' should be a float greater or equal than 0.")

        if not (env_map == "" or Utils.get_uaibot_type(env_map) == "uaibot.Texture"):
            env_map = Texture(env_map)

        if not Utils.is_a_number(env_map_intensity) or env_map_intensity < 0:
            raise Exception("The parameter 'env_map_intensity' should be a float greater or equal than 0.")

        if not (str(type(flat_shading)) == "<class 'bool'>"):
            raise Exception("The parameter '_flat_shading' should be a boolean.")

        if not (light_map == "" or Utils.get_uaibot_type(light_map) == "uaibot.Texture"):
            light_map = Texture(light_map)

        if not Utils.is_a_number(light_map_itensity) or light_map_itensity < 0:
            raise Exception("The parameter 'light_map_itensity' should be a float greater or equal than 0.")

        if not (texture_map == "" or Utils.get_uaibot_type(texture_map) == "uaibot.Texture"):
            texture_map = Texture(texture_map)

        if not Utils.is_a_number(metalness) or metalness < 0 or metalness > 1:
            raise Exception("The parameter 'metalness' should be a float between 0 (non-metal) and 1 (metal).")

        if not (metalness_map == "" or Utils.get_uaibot_type(metalness_map) == "uaibot.Texture"):
            metalness_map = Texture(metalness_map)

        if not (normal_map == "" or Utils.get_uaibot_type(normal_map) == "uaibot.Texture"):
            normal_map = Texture(normal_map)

        if not Utils.is_a_vector(normal_scale, 2):
            raise Exception("The parameter 'normal_scale' should be a 2D vector.")

        if not Utils.is_a_number(refraction_ratio) or refraction_ratio < 0 or refraction_ratio > 1:
            raise Exception("The parameter 'refraction_ratio' should be a float between 0 and 1.")

        if not Utils.is_a_number(roughness) or roughness < 0 or roughness > 1:
            raise Exception("The parameter 'roughness' should be a float between 0 (mirror) and 1 (fully diffuse).")

        if not (roughness_map == "" or Utils.get_uaibot_type(roughness_map) == "uaibot.Texture"):
            roughness_map = Texture(roughness_map)

        if not Utils.is_a_number(clearcoat) or clearcoat < 0 or clearcoat > 1:
            raise Exception("The parameter 'clearcoat' should be a float between 0 and 1.")

        if not (clearcoat_map == "" or Utils.get_uaibot_type(clearcoat_map) == "uaibot.Texture"):
            clearcoat_map = Texture(clearcoat_map)

        if not (clearcoat_normal_map == "" or Utils.get_uaibot_type(clearcoat_normal_map) == "uaibot.Texture"):
            clearcoat_normal_map = Texture(clearcoat_normal_map)

        if not Utils.is_a_vector(clearcoat_normal_scale, 2):
            raise Exception("The parameter 'clearcoat_normal_scale' should be a 2D vector.")

        if not Utils.is_a_number(clearcoat_roughness) or clearcoat_roughness < 0 or clearcoat_roughness > 1:
            raise Exception("The parameter 'clearcoat_roughness' should be a float between 0 and 1.")

        if not (clearcoat_roughness_map == "" or Utils.get_uaibot_type(
                clearcoat_roughness_map) == "uaibot.Texture"):
            clearcoat_roughness_map = Texture(clearcoat_roughness_map)

        if not Utils.is_a_number(ior) or ior < 1 or ior > 2.333:
            raise Exception("The parameter 'ior' should be a float between 1 and 2.333.")

        if not Utils.is_a_number(reflectivity) or reflectivity < 0 or reflectivity > 1:
            raise Exception("The parameter 'reflectivity' should be a float between 0 and 1.")

        if not Utils.is_a_number(sheen) or sheen < 0 or sheen > 1:
            raise Exception("The parameter 'sheen' should be a float between 0 and 1.")

        if not Utils.is_a_number(sheen_roughness) or sheen_roughness < 0 or sheen_roughness > 1:
            raise Exception("The parameter 'sheen_roughness' should be a float between 0 and 1.")

        if not (sheen_roughness_map == "" or Utils.get_uaibot_type(sheen_roughness_map) == "uaibot.Texture"):
            sheen_roughness_map = Texture(sheen_roughness_map)

        if not Utils.is_a_color(sheen_color):
            raise Exception("The parameter 'sheen_color' should be a HTML-style color.")

        if not (sheen_color_map == "" or Utils.get_uaibot_type(sheen_color_map) == "uaibot.Texture"):
            sheen_color_map = Texture(sheen_color_map)

        if not Utils.is_a_number(specular_intensity) or specular_intensity < 0 or specular_intensity > 1:
            raise Exception("The parameter 'specular_intensity' should be a float between 0 and 1.")

        if not (specular_intensity_map == "" or Utils.get_uaibot_type(
                specular_intensity_map) == "uaibot.Texture"):
            specular_intensity_map = Texture(specular_intensity_map)

        if not Utils.is_a_color(specular_color):
            raise Exception("The parameter 'specular_color' should be a HTML-style color.")

        if not (specular_color_map == "" or Utils.get_uaibot_type(specular_color_map) == "uaibot.Texture"):
            specular_color_map = Texture(specular_color_map)

        if not Utils.is_a_number(transmission) or transmission < 0 or transmission > 1:
            raise Exception("The parameter 'transmission' should be a float between 0 and 1.")

        if not (transmission_map == "" or Utils.get_uaibot_type(transmission_map) == "uaibot.Texture"):
            transmission_map = Texture(transmission_map)

        if transmission > 0 and opacity < 1:
            raise Exception("If 'transmission' is non-zero, 'opacity' should be set to 1.")
        # Code

        self._opacity = opacity
        self._shadow_side = shadow_side
        self._side = side

        self._transparent = self._opacity < 1
        self._alpha_map = alpha_map
        self._ao_map = ao_map
        self._ao_map_intensity = ao_map_intensity
        self._bump_map = bump_map
        self._bump_map_scale = bump_map_scale
        self._color = color
        self._emissive = emissive
        self._emissive_map = emissive_map
        self._emissive_intensity = emissive_intensity
        self._env_map = env_map
        self._env_map_intensity = env_map_intensity
        self._flat_shading = flat_shading
        self._light_map = light_map
        self._light_map_itensity = light_map_itensity
        self._map = texture_map
        self._metalness = metalness
        self._metalness_map = metalness_map
        self._normal_map = normal_map
        self._normal_scale = normal_scale
        self._refraction_ratio = refraction_ratio
        self._roughness = roughness
        self._roughness_map = roughness_map

        self._clearcoat = clearcoat
        self._clearcoat_map = clearcoat_map
        self._clearcoat_normal_map = clearcoat_normal_map
        self._clearcoat_normal_scale = clearcoat_normal_scale
        self._clearcoat_roughness = clearcoat_roughness
        self._clearcoat_roughness_map = clearcoat_roughness_map
        self._ior = ior
        self._reflectivity = reflectivity
        self._sheen = sheen
        self._sheen_roughness = sheen_roughness
        self._sheen_roughness_map = sheen_roughness_map
        self._sheen_color = sheen_color
        self._sheen_color_map = sheen_color_map
        self._specular_intensity = specular_intensity
        self._specular_intensity_map = specular_intensity_map
        self._specular_color = specular_color
        self._specular_color_map = specular_color_map
        self._transmission = transmission
        self._transmission_map = transmission_map

    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Mesh Material"

        return string

    #######################################
    # Methods
    #######################################


    @staticmethod
    def create_glass(color : Optional[str] = None)-> "MeshMaterial":
        
        if color is None:
            return MeshMaterial(metalness = 0.9, roughness = 0.05, clearcoat = 1,
                            reflectivity = 0.2,  env_map_intensity=  0.9, side='DoubleSide',
                            refraction_ratio= 0.985, ior = 1, opacity=0.5)
        else:
            return MeshMaterial(metalness = 0.9, roughness = 0.05, clearcoat = 1,
                            reflectivity = 0.2,  env_map_intensity=  0.9, side='DoubleSide',
                            refraction_ratio= 0.985, ior = 1, opacity=0.5, color = color)            
 
    @staticmethod
    def create_colored_metal(color : str = 'white')-> "MeshMaterial":
   
        texture = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/white.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

        return MeshMaterial(metalness=0.9, clearcoat=0.5, roughness=0.0, 
                            flat_shading=True, normal_scale=[1, 1], 
                            texture_map=texture, color=color, side='DoubleSide')


    @staticmethod
    def create_gold()-> "MeshMaterial":
   
        texture = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/gold_metal.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

        return MeshMaterial(metalness=0.9, clearcoat=0.5, roughness=0.0, 
                            flat_shading=True, normal_scale=[1, 1], 
                            texture_map=texture, side='DoubleSide')

    @staticmethod
    def create_wood()-> "MeshMaterial":
   
        texture = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/wood_1.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

        return MeshMaterial(metalness=0, roughness=1.0, 
                            flat_shading=True, normal_scale=[1, 1], 
                            texture_map=texture, side='DoubleSide')

    @staticmethod
    def create_rough_metal()-> "MeshMaterial":
   
        texture = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/rough_metal.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

        return MeshMaterial(metalness=0.5, roughness=0.5, 
                            flat_shading=True, normal_scale=[1, 1], 
                            texture_map=texture, side='DoubleSide')
                                                 
    def gen_code(self, name, port):

        string = ""

        # Create textures
        if not (self.map == ""):
            string += self.map.gen_code("map_" + name, port = port)

        if not (self.alpha_map == ""):
            string += self.alpha_map.gen_code("alpha_map_" + name, port = port)

        if not (self.ao_map == ""):
            string += self.ao_map.gen_code("ao_map_" + name, port = port)

        if not (self.bump_map == ""):
            string += self.bump_map.gen_code("bump_map_" + name, port = port)

        if not (self.emissive_map == ""):
            string += self.emissive_map.gen_code("emissive_map_" + name, port = port)

        if not (self.env_map == ""):
            string += self.env_map.gen_code("env_map_" + name, port = port)

        if not (self.light_map == ""):
            string += self.light_map.gen_code("light_map_" + name, port = port)

        if not (self.metalness_map == ""):
            string += self.metalness_map.gen_code("metalness_map_" + name, port = port)

        if not (self.normal_map == ""):
            string += self.normal_map.gen_code("normal_map_" + name, port = port)

        if not (self.roughness_map == ""):
            string += self.roughness_map.gen_code("roughness_map_" + name, port = port)

        if not (self.clearcoat_map == ""):
            string += self.clearcoat_map.gen_code("clearcoat_map_" + name, port = port)

        if not (self.clearcoat_normal_map == ""):
            string += self.clearcoat_normal_map.gen_code("clearcoat_normal_map_" + name, port = port)

        if not (self.clearcoat_roughness_map == ""):
            string += self.clearcoat_roughness_map.gen_code("clearcoat_roughness_map_" + name, port = port)

        if not (self.sheen_roughness_map == ""):
            string += self.sheen_roughness_map.gen_code("sheen_roughness_map_" + name, port = port)

        if not (self.sheen_color_map == ""):
            string += self.sheen_color_map.gen_code("sheen_color_map_" + name, port = port)

        if not (self.specular_intensity_map == ""):
            string += self.specular_intensity_map.gen_code("specular_intensity_map_" + name, port = port)

        if not (self.specular_color_map == ""):
            string += self.specular_color_map.gen_code("specular_color_map_" + name, port = port)

        if not (self.transmission_map == ""):
            string += self.transmission_map.gen_code("transmission_map_" + name, port = port)

        # Create mesh
        string += "const material_" + name + " = new MeshPhysicalMaterial({\n"

        string += "opacity: " + str(self.opacity) + ",\n"

        if not (self.shadow_side == "null"):
            string += "shadowSide: " + self.shadow_side + ",\n"

        if not (self.side == "null"):
            string += "side: " + self.side + ",\n"

        string += "transparent: " + str(self.transparent).lower() + ",\n"

        if not (self.alpha_map == ""):
            string += "alphaMap: texture_alpha_map_" + name + ",\n"

        if not (self.ao_map == ""):
            string += "aoMap: texture_alpha_map_" + name + ",\n"

        string += "aoMapIntensity: " + str(self.ao_map_intensity) + ",\n"

        if not (self.bump_map == ""):
            string += "bumpMap: texture_bump_map_" + name + ",\n"

        string += "bumpScale: " + str(self.bump_map_scale) + ",\n"
        string += "color: \"" + self.color + "\",\n"
        string += "emissive: \"" + self.emissive + "\",\n"

        if not (self.emissive_map == ""):
            string += "emissiveMap: texture_emissive_map_" + name + ",\n"

        string += "emissiveIntensity: " + str(self.emissive_intensity) + ",\n"

        if not (self.env_map == ""):
            string += "envMap: texture_env_map_" + name + ",\n"

        string += "envMapIntensity: " + str(self.env_map_intensity) + ",\n"
        string += "flatShading: " + str(self.flat_shading).lower() + ",\n"

        if not (self.light_map == ""):
            string += "lightMap: texture_light_map_" + name + ",\n"

        string += "lightMapIntensity: " + str(self.light_map_itensity) + ",\n"

        if not (self.map == ""):
            string += "map: texture_map_" + name + ",\n"

        string += "metalness: " + str(self.metalness) + ",\n"

        if not (self.metalness_map == ""):
            string += "metalnessMap: texture_metalness_map_" + name + ",\n"

        if not (self.normal_map == ""):
            string += "normalMap: texture_normal_map_" + name + ",\n"

        string += "normalScale: new Vector2(" + str(self.normal_scale[0]) + "," + str(self.normal_scale[1]) + "),\n"
        string += "refractionRatio: " + str(self.refraction_ratio) + ",\n"
        string += "roughness: " + str(self.roughness) + ",\n"

        if not (self.roughness_map == ""):
            string += "roughnessMap: texture_roughness_map_" + name + ",\n"

        string += "clearcoat: " + str(self.clearcoat) + ",\n"

        if not (self.clearcoat_map == ""):
            string += "clearcoatMap: texture_clearcoat_map_" + name + ",\n"

        if not (self.clearcoat_normal_map == ""):
            string += "clearcoatNormalMap: texture_clearcoat_normal_map_" + name + ",\n"

        string += "clearcoatNormalScale: new Vector2(" + str(self.clearcoat_normal_scale[0]) + "," + str(
            self.clearcoat_normal_scale[1]) + "),\n"
        string += "clearcoatRoughness: " + str(self.clearcoat_roughness) + ",\n"

        if not (self.clearcoat_roughness_map == ""):
            string += "clearcoatRoughnessMap: texture_clearcoat_roughness_map_" + name + ",\n"

        string += "ior: " + str(self.ior) + ",\n"
        string += "reflectivity: " + str(self.reflectivity) + ",\n"
        string += "sheen: " + str(self.sheen) + ",\n"
        string += "sheenRoughness: " + str(self.sheen_roughness) + ",\n"

        if not (self.sheen_roughness_map == ""):
            string += "sheenRoughnessMap: texture_sheen_roughness_map_" + name + ",\n"

        string += "sheenColor: \"" + self.sheen_color + "\",\n"

        if not (self.sheen_color_map == ""):
            string += "sheenColorMap: texture_sheen_color_map" + name + ",\n"

        string += "specularIntensity: " + str(self.specular_intensity) + ",\n"

        if not (self.specular_intensity_map == ""):
            string += "specularIntensityMap: texture_specular_intensity_map_" + name + ",\n"

        string += "specularColor: \"" + self._specular_color + "\",\n"

        if not (self.specular_color_map == ""):
            string += "specularColorMap: texture_specular_color_map_" + name + ",\n"

        string += "transmission: " + str(self._transmission) + ",\n"

        if not (self.transmission_map == ""):
            string += "transmissionMap: texture_transmission_map_" + name + ",\n"

        string += "});\n\n"

        return string
