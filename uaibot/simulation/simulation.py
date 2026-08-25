try:
    # New versions
    from IPython.display import display, HTML
except ImportError:
    # Fallback for old versions
    from IPython.core.display import display, HTML
import re
from utils import *
from graphics.meshmaterial import *
from simobjects.box import *
from simobjects.pointlight import *
from simobjects.pointcloud import *
from simobjects.frame import *
import time
from pathlib import Path
import httplib2
import sys
from uaibot.utils.types import HTMatrix, Matrix, Vector, SimObject
from typing import Optional, Tuple, List
import os
import inspect
import webbrowser
import random
import tempfile

class Simulation:
    """
  A simulation variable.

  Parameters
  ----------
  obj_list : A list of objects that can be simulated (see Utils.IS_OBJ_SIM)
      The objects that will be added initially to the simulation.
      (default: empty list)

  ambient_light_intensity : float
      The intensity of the ambient light.
      (default: 12).

  ldr_urls : a list of six url strings or None
      A list containing the LDR lightning images in the following order:
      [positive_x, negative_x, positive_y, negative_y, positive_z, negative_z].
      If None, no LDR is used.
      (default: None).

  camera_type : string
      The camera type, either "orthographic" or "perspective".
      (default: "perspective").

  width : positive float
      The canvas width, in pixels.
      (default: [] (automatic)).

  height : positive float
      The canvas height, in pixels.
      (default: [] (automatic)).

  show_world_frame: boolean
      If the frame in the middle of the scenario is shown.
      (default: True).

  show_grid : boolean
      If the grid in the scenario is shown.
      (default: True).

  load_screen_color : string, a HTML-compatible color
      The color of the loading screen.
      (default: "#19bd39").

  background_color : string, a HTML-compatible color
      The color of the background.
      (default: "white").

  camera_start_pose: vector or list with 7 entries, or None
      The camera starting configuration. The first three elements is the camera position (x,y,z).
      The next three is a point in which the camera is looking at.
      The final one is the camera zoom.
      If None, uses a default configuration for the camera.
      (default: None).
      
  pixel_ratio: float between 0 and 1,
      A parameter that sets the resolution. Can be reduced from 1 to reduce the rendering load.
      (default: 0.9).
    
  anti_aliasing: boolean
      If anti_aliasing is used in rendering or not.
      (default: True).
      
  local_host_port: string
      If you want the data (3D models, textures, etc...) of your animation to 
      come from a localhost, specify the port here as a string (e.g. '8000').
      If empty (''), it comes from an online server.
      (default: '')
          
  """

    _CAMERATYPE = ['perspective', 'orthographic']
    # Import the javascript code as a string



    #######################################
    # Attributes
    #######################################

    @property
    def list_of_objects(self) -> List["SimObject"]:
        """A list of all sim objects."""
        return self._list_of_objects

    @property
    def list_of_names(self) -> List[str]:
        """A list of all object names."""
        return self._list_of_names

    @property
    def ambient_light_intensity(self) -> float:
        """Ambient light intensity."""
        return self._ambient_light_intensity

    @property
    def ldr_urls(self) -> List[str]:
        """A list of the LDR light urls."""
        return self._ldr_urls

    @property
    def camera_type(self) -> str:
        """Type of the camera."""
        return self._camera_type

    @property
    def width(self) -> int:
        """Width, in pixels, of the canvas"""
        return self._width

    @property
    def height(self) -> int:
        """Height, in pixels, of the canvas"""
        return self._height

    @property
    def show_world_frame(self) -> bool:
        """If the world frame is shown"""
        return self._show_world_frame

    @property
    def show_grid(self) -> bool:
        """If the grid in the world is shown"""
        return self._show_grid

    @property
    def load_screen_color(self) -> str:
        """Loading screen color"""
        return self._load_screen_color

    @property
    def background_color(self) -> str:
        """Color of the background of the scenario"""
        return self._background_color

    @property
    def camera_start_pose(self) -> List[float]:
        """The camera starting pose. The first three elements are the starting camera position, the next three ones
        is the starting point in which the camera is looking at and the last one is the zoom"""
        return self._camera_start_pose
    
    @property
    def pixel_ratio(self) -> float:
        """Relate to the resolution. Make this number smaller to ahve a performance boost"""
        return self._pixel_ratio

    @property
    def anti_aliasing(self) -> bool:
        """If the animation uses anti aliasing. Make this number smaller to ahve a performance boost"""
        return self._anti_aliasing
    
    @property
    def local_host_port(self) -> str:
        """Port of your local host serving the models."""
        return self._local_host_port
    
    #######################################
    # Constructor
    #######################################

    def __init__(self, obj_list: List["SimObject"]=[], ambient_light_intensity: float =4, 
                 ldr_urls: Optional[List[str]] = None, camera_type: str ="perspective", 
                 width: Optional[int] =[], height: Optional[int] = [], 
                 show_world_frame: bool = True, show_grid: bool = True, 
                 load_screen_color: str ="#19bd39", background_color: str ="#F5F6FA",
                 camera_start_pose: Optional[List[float]] = None,
                 pixel_ratio : float = 0.9, anti_aliasing: bool = True,
                 local_host_port: str = '') -> "Simulation":

        if not Utils.is_a_number(ambient_light_intensity) or ambient_light_intensity < 0:
            raise Exception("The parameter 'ambient_light_intensity' should be a nonnegative float.")

        if not (camera_type in Simulation._CAMERATYPE):
            raise Exception("The parameter 'camera_type' must be one of the following strings: " + str(
                Simulation._CAMERATYPE) + ".")

        
        if (not width==[]) and (not Utils.is_a_number(width) or width <= 0):
            raise Exception("The parameter 'width' must be a positive float or [] (automatic).")

        if (not height==[]) and (not Utils.is_a_number(height) or height <= 0):
            raise Exception("The parameter 'height' must be a positive float or [] (automatic).")
        
        if not str(type(show_world_frame)) == "<class 'bool'>":
            raise Exception("The parameter 'show_world_frame' must be a boolean.")

        if not str(type(show_grid)) == "<class 'bool'>":
            raise Exception("The parameter 'show_grid' must be a boolean.")

        if not Utils.is_a_color(load_screen_color):
            raise Exception("The parameter 'load_screen_color' must be a HTML-compatible color.")

        if not Utils.is_a_color(background_color):
            raise Exception("The parameter 'background_color' must be a HTML-compatible color.")


        if not isinstance(local_host_port, str):
            raise Exception("The parameter 'local_host_port' must be a string.")
        

        if camera_start_pose is None:
            if camera_type=="perspective":
                camera_start_pose = [1.76, 1.10, 1.45, 0, 0, 0, 1]
            else:
                camera_start_pose = [1.3, 1.8, 2.7, 0, 0, 0, 4]

        if not Utils.is_a_vector(camera_start_pose,7):
            raise Exception("The parameter 'camera_start_pose' should be either None or a 7 element vector.")

        if not Utils.is_a_number(pixel_ratio) or pixel_ratio < 0 or pixel_ratio>1:
            raise Exception("The parameter 'pixel_ratio' should be a float number between 0 and 1.")

        if not str(type(anti_aliasing)) == "<class 'bool'>":
            raise Exception("The parameter 'anti_aliasing' must be a boolean.")
                
        self._list_of_objects = []
        self._list_of_names = []
        self._ambient_light_intensity = ambient_light_intensity
        self._camera_type = camera_type
        self._ldr_urls = ldr_urls
        
        if width==[] or height==[]:
            if Utils.get_environment() == 'Local':
                self._width = 960 
                self._height = 720 
            else:
                self._width = 500
                self._height = 500                
        else:
            self._width = width
            self._height = height
            
        self._show_world_frame = show_world_frame
        self._show_grid = show_grid
        self._load_screen_color = load_screen_color
        self._background_color = background_color
        self._camera_start_pose = np.array(camera_start_pose).tolist()
        self._pixel_ratio = pixel_ratio
        self._anti_aliasing = anti_aliasing
        self._local_host_port = local_host_port   
             
        #Add reference frame
        if self._show_world_frame:
            self.add(Frame(name='w0_frame', htm=Utils.trn([0,0,0]),size=0.5))
        
        self.add(obj_list)


    #######################################
    # Std. Print
    #######################################

    def __repr__(self):

        string = "Simulation: \n\n"
        string += " Variables: \n"
        string += str(self.list_of_names)
        string += " Width: "+str(self.width)+"px, Height: "+str(self.height)+"px\n"
        string += " Camera type: "+str(self.camera_type)+"\n"
        string += " Ambient light intensity: " + str(self.ambient_light_intensity) + "\n"
        string += " Show world frame: " + str(self.show_world_frame) + "\n"
        string += " Show grid: " + str(self.show_grid) + "\n"
        string += " Background color: " + str(self.background_color) + "\n"

        return string

    #######################################
    # Methods
    #######################################

    @staticmethod
    def create_sim_factory(objects: List["SimObject"] =[], light_intensity: float = 1.0) -> "Simulation":
        """
    Create an environment of a factory.
    Factory panorama taken from:
    'https://www.samrohn.com/360-panorama/chrysler-factory-detroit-usa-360-tour/chrysler-factory-360-panorama-tour-007/'

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """

        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/factory_ground.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])

        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=1, roughness=1)

        ground = Box(name="ground", width=6, depth=6, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-1,-1, 1.5]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-1, 1, 1.5]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 1,-1, 1.5]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 1, 1, 1.5]))

        ldr_url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/LDR/factory_"
        ldr_list = [ldr_url + "px.png", ldr_url + "nx.png", ldr_url + "py.png", ldr_url + "ny.png", ldr_url + "nz.png",
                    ldr_url + "nz.png"]

        sim = Simulation(objects, ambient_light_intensity=2*light_intensity, ldr_urls=ldr_list)
        sim.add(ground)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)

        return sim


    @staticmethod
    def create_sim_sky(objects: List["SimObject"] =[], light_intensity: float = 1.0) -> "Simulation":
        """
    Create an environment of a sky.
    Outside panorama taken from:
    'https://polyhaven.com/a/citrus_orchard_puresky'

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """

        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/grass.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[100, 100])

        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=0, roughness=1)

        ground = Box(name="ground", width=100, depth=100, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-1,-1, 1.5]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-1, 1, 1.5]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 1,-1, 1.5]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 1, 1, 1.5]))

        ldr_url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/LDR/puresky_"
        ldr_list = [ldr_url + "px.png", ldr_url + "nx.png", ldr_url + "py.png", ldr_url + "ny.png", ldr_url + "pz.png",
                    ldr_url + "nz.png"]


        sim = Simulation(objects, ambient_light_intensity=2*light_intensity, ldr_urls=ldr_list)
        sim.add(ground)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)

        return sim
    
    @staticmethod
    def create_sim_snow(objects: List["SimObject"] =[], light_intensity: float = 1.0) -> "Simulation":
        """
    Create an environment of a snowy region.
    Outside panorama taken from:
    'https://polyhaven.com/a/passendorf_snow'

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """

        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/snow.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[20, 20])

        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=0, roughness=1)

        ground = Box(name="ground", width=100, depth=100, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-1,-1, 1.5]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-1, 1, 1.5]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 1,-1, 1.5]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 1, 1, 1.5]))

        ldr_url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/LDR/snow_"
        ldr_list = [ldr_url + "px.png", ldr_url + "nx.png", ldr_url + "py.png", ldr_url + "ny.png", ldr_url + "pz.png",
                    ldr_url + "nz.png"]


        sim = Simulation(objects, ambient_light_intensity=2*light_intensity, ldr_urls=ldr_list)
        sim.add(ground)


        return sim
            
    @staticmethod
    def create_sim_mountain(objects: List["SimObject"] =[], light_intensity: float = 1.0) -> "Simulation":
        """
    Create an environment of a mountain.
    Outside panorama taken from:
    'https://opengameart.org/content/skiingpenguins-skybox-pack'

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """

        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/grass.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[100, 100])

        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=0, roughness=1)

        ground = Box(name="ground", width=100, depth=100, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-1,-1, 1.5]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-1, 1, 1.5]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 1,-1, 1.5]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 1, 1, 1.5]))

        ldr_url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/LDR/green_"
        ldr_list = [ldr_url + "px.jpg", ldr_url + "nx.jpg", ldr_url + "py.jpg", ldr_url + "ny.jpg", ldr_url + "pz.jpg",
                    ldr_url + "nz.jpg"]


        sim = Simulation(objects, ambient_light_intensity=2*light_intensity, ldr_urls=ldr_list)
        sim.add(ground)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)

        return sim

    @staticmethod
    def create_sim_hill(objects: List["SimObject"] =[], light_intensity: float = 1.0) -> "Simulation":
        """
    Create an environment of a hill.
    Outside panorama taken from:
    'https://polyhaven.com/a/spaichingen_hill'

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """


        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/grass.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[20, 20])

        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=0, roughness=1)

        ground = Box(name="ground", width=20, depth=20, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-1,-1, 1.5]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-1, 1, 1.5]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 1,-1, 1.5]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 1, 1, 1.5]))

        ldr_url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/LDR/hill_"
        ldr_list = [ldr_url + "px.png", ldr_url + "nx.png", ldr_url + "py.png", ldr_url + "ny.png", ldr_url + "pz.png",
                    ldr_url + "nz.png"]


        sim = Simulation(objects, ambient_light_intensity=2*light_intensity, ldr_urls=ldr_list)
        sim.add(ground)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)

        return sim

    @staticmethod
    def create_sim_orchard_road(objects: List["SimObject"] =[], light_intensity: float = 1.0) -> "Simulation":
        """
    Create an environment of an orchard road.
    Outside panorama taken from:
    'https://polyhaven.com/a/citrus_orchard_road'

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """

        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/ground_orchard.jpg',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[5, 5])

        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=0, roughness=1)

        ground = Box(name="ground", width=20, depth=20, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-1,-1, 1.5]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-1, 1, 1.5]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 1,-1, 1.5]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 1, 1, 1.5]))

        ldr_url = "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/LDR/orchard_road_"
        ldr_list = [ldr_url + "px.png", ldr_url + "nx.png", ldr_url + "py.png", ldr_url + "ny.png", ldr_url + "pz.png",
                    ldr_url + "nz.png"]


        sim = Simulation(objects, ambient_light_intensity=2*light_intensity, ldr_urls=ldr_list)
        sim.add(ground)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)

        return sim

    @staticmethod
    def create_sim_grid(objects: List["SimObject"] =[], light_intensity: float = 3.0) -> "Simulation":
        """
    Create an environment of a grid (dark).

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """


        
        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/grid_dark.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])
        
        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=0, roughness=1)
        
        size = 20

        ground = Box(width=size, depth=size, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        wall1 = Box(width=0.005, depth=size, height=size, htm=Utils.trn([size/2, 0, size/2]),
                     mesh_material=mesh_ground)

        wall2 = Box(width=0.005, depth=size, height=size, htm=Utils.trn([-size/2, 0, size/2]),
                     mesh_material=mesh_ground)

        wall3 = Box(width=size, depth=0.005, height=size, htm=Utils.trn([0, size/2, size/2]),
                     mesh_material=mesh_ground)

        wall4 = Box(width=size, depth=0.005, height=size, htm=Utils.trn([0, -size/2, size/2]),
                     mesh_material=mesh_ground)
                        
        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-2,-2, 5.0]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-2, 2, 5.0]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 2,-2, 5.0]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 2, 2, 5.0]))
        light5 = PointLight(name="light5", color="white", intensity=light_intensity, htm=Utils.trn([ 0,0,5]))

        sim = Simulation(objects, ambient_light_intensity=light_intensity, show_grid=False)
        sim.add(ground)
        sim.add(wall1)
        sim.add(wall2)
        sim.add(wall3)
        sim.add(wall4)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)
        sim.add(light5)

        return sim

    @staticmethod
    def create_sim_grid_blue(objects: List["SimObject"] =[], light_intensity: float = 2.0) -> "Simulation":
        """
    Create an environment of a grid (blue).

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """


        
        texture_ground = Texture(
            url='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/Textures/grid.png',
            wrap_s='RepeatWrapping', wrap_t='RepeatWrapping', repeat=[4, 4])
        
        mesh_ground = MeshMaterial(texture_map=texture_ground, metalness=0, roughness=0)
        
        size = 20

        ground = Box(width=size, depth=size, height=0.01, htm=Utils.trn([0, 0, -0.005]),
                     mesh_material=mesh_ground)

        wall1 = Box(width=0.005, depth=size, height=size, htm=Utils.trn([size/2, 0, size/2]),
                     mesh_material=mesh_ground)

        wall2 = Box(width=0.005, depth=size, height=size, htm=Utils.trn([-size/2, 0, size/2]),
                     mesh_material=mesh_ground)

        wall3 = Box(width=size, depth=0.005, height=size, htm=Utils.trn([0, size/2, size/2]),
                     mesh_material=mesh_ground)

        wall4 = Box(width=size, depth=0.005, height=size, htm=Utils.trn([0, -size/2, size/2]),
                     mesh_material=mesh_ground)
                        
        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-2,-2, 5.0]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-2, 2, 5.0]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 2,-2, 5.0]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 2, 2, 5.0]))
        light5 = PointLight(name="light5", color="white", intensity=light_intensity, htm=Utils.trn([ 0,0,5]))

        sim = Simulation(objects, ambient_light_intensity=light_intensity, show_grid=False)
        sim.add(ground)
        sim.add(wall1)
        sim.add(wall2)
        sim.add(wall3)
        sim.add(wall4)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)
        sim.add(light5)

        return sim
                
    @staticmethod
    def create_sim_lesson(objects: List["SimObject"] =[], light_intensity: float = 1.0) -> "Simulation":
        """
    Create an environment for embedding into the lessons slides.

    Parameters
    ----------
    objects: list of objects that can be simulated (see Utils.IS_OBJ_SIM)
        The objects to be added to the scenario.

    Returns
    -------
    sim: 'Simulation' object
        Simulation object.
    """


                            
        light1 = PointLight(name="light1", color="white", intensity=light_intensity, htm=Utils.trn([-1,-1, 1.0]))
        light2 = PointLight(name="light2", color="white", intensity=light_intensity, htm=Utils.trn([-1, 1, 1.0]))
        light3 = PointLight(name="light3", color="white", intensity=light_intensity, htm=Utils.trn([ 1,-1, 1.0]))
        light4 = PointLight(name="light4", color="white", intensity=light_intensity, htm=Utils.trn([ 1, 1, 1.0]))
        light5 = PointLight(name="light5", color="white", intensity=light_intensity, htm=Utils.trn([ 0,0,5]))


        sim = Simulation(objects, ambient_light_intensity=2*light_intensity, show_grid=False)
        sim.add(light1)
        sim.add(light2)
        sim.add(light3)
        sim.add(light4)
        sim.add(light5)

        return sim

    def set_parameters(self, ambient_light_intensity: Optional[float]= None, ldr_urls: Optional[str] =None, 
                       camera_type: Optional[str] =None, width: Optional[int] =None,
                 height: Optional[int] = None, show_world_frame: Optional[bool] = None, 
                 show_grid: Optional[bool] = None, load_screen_color: Optional[str] = None, 
                 background_color: Optional[str] = None, camera_start_pose: Optional[List[float]] = None,
                 pixel_ratio : Optional[float] = None, anti_aliasing: Optional[bool] = None, 
                 local_host_port: Optional[str] = None) -> None:
        """
      Change the simulation parameters.

      Parameters
      ----------
      ambient_light_intensity : float
          The intensity of the ambient light.
          If None, does not change the current value.
          (default: None).

      ldr_urls : a list of six url strings or None
          A list containing the LDR lightning images in the following order:
          [positive_x, negative_x, positive_y, negative_y, positive_z, negative_z].
          If None, does not change the current value.
          (default: None).

      camera_type : string
          The camera type, either "orthographic" or "perspective".
          If None, does not change the current value.
          (default: None).

      width : positive float
          The canvas width, in pixels.
          If None, does not change the current value.
          (default: None).

      height : positive float
          The canvas height, in pixels.
          If None, does not change the current value.
          (default: None).

      show_world_frame: boolean
          If the frame in the middle of the scenario is shown.
          If None, does not change the current value.
          (default: None).

      show_grid : boolean
          If the grid in the scenario is shown.
          If None, does not change the current value.
          (default: None).

      load_screen_color : string, a HTML-compatible color
          The color of the loading screen.
          If None, does not change the current value.
          (default: None).

      background_color : string, a HTML-compatible color
          The color of the background.
          If None, does not change the current value.
          (default: None).

      camera_start_pose: vector or list with 7 entries, or None
          The camera starting configuration. The first three elements is the camera position (x,y,z).
          The next three is a point in which the camera is looking at.
          The final one is the camera zoom.
          If None, does not change the current value.
          (default: None).
          
      pixel_ratio: float between 0 and 1, or None
          A parameter that sets the resolution. Can be reduced from 1 to reduce the rendering load.
          (default: None).
          
      anti_aliasing: boolean, or None
          If anti_aliasing is used in rendering or not.
          (default: None).
      """

        if (not ambient_light_intensity is None) and (not Utils.is_a_number(ambient_light_intensity) or ambient_light_intensity < 0):
            raise Exception("The parameter 'ambient_light_intensity' should be a nonnegative float.")

        if (not camera_type is None) and  (not (camera_type in Simulation._CAMERATYPE) or (not ambient_light_intensity is None)):
            raise Exception("The parameter 'camera_type' must be one of the following strings: " + str(
                Simulation._CAMERATYPE) + ".")

        if (not width is None) and  (not Utils.is_a_number(width) or width <= 0):
            raise Exception("The parameter 'width' must be a positive float.")

        if (not height is None) and  (not Utils.is_a_number(height) or height <= 0):
            raise Exception("The parameter 'height' must be a positive float.")

        if (not show_world_frame is None) and (not str(type(show_world_frame)) == "<class 'bool'>"):
            raise Exception("The parameter 'show_world_frame' must be a boolean.")

        if (not show_grid is None) and (not str(type(show_grid)) == "<class 'bool'>"):
            raise Exception("The parameter 'show_grid' must be a boolean.")

        if (not load_screen_color is None) and (not Utils.is_a_color(load_screen_color)):
            raise Exception("The parameter 'load_screen_color' must be a HTML-compatible color.")

        if (not background_color is None) and  (not Utils.is_a_color(background_color)):
            raise Exception("The parameter 'background_color' must be a HTML-compatible color.")

        if not (ldr_urls is None):
            if not (str(type(ldr_urls)) == "<class 'list'>") or not (len(ldr_urls) == 6):
                raise Exception("The parameter 'ldr_urls' should be a list of six urls or 'None'.")

        if (not camera_start_pose is None) and (not Utils.is_a_vector(camera_start_pose,7)):
            raise Exception("The parameter 'camera_start_pose' should be either None or a 7 element vector.")

        if (not pixel_ratio is None) and (not Utils.is_a_number(pixel_ratio) or pixel_ratio < 0 or pixel_ratio>1):
            raise Exception("The parameter 'pixel_ratio' should be a float number between 0 and 1.")

        if (not anti_aliasing is None) and (not str(type(anti_aliasing)) == "<class 'bool'>"):
            raise Exception("The parameter 'anti_aliasing' must be a boolean.")
        
        if (not local_host_port is None) and (not isinstance(local_host_port, str)):
            raise Exception("The parameter 'local_host_port' must be a string.")
        
        
        
        if not ambient_light_intensity is None:
            self._ambient_light_intensity = ambient_light_intensity

        if not camera_type is None:
            self._camera_type = camera_type

        if not ldr_urls is None:
            self._ldr_urls = ldr_urls

        if not width is None:
            self._width = width

        if not height is None:
            self._height = height

        if not show_world_frame is None:
            if show_world_frame and (not self._show_world_frame):
                self.add(Frame(name='w0_frame', htm=Utils.trn([0,0,0]),size=0.5))
            if (not show_world_frame) and self._show_world_frame:
                for j in range(len(self._list_of_names)):
                    if self._list_of_names[j] == 'w0_frame':
                        j_remove=j
                    
                del self._list_of_names[j_remove]
                del self._list_of_objects[j_remove]
                    
            self._show_world_frame = show_world_frame

        if not show_grid is None:
            self._show_grid = show_grid

        if not load_screen_color is None:
            self._load_screen_color = load_screen_color

        if not background_color is None:
            self._background_color = background_color

        if not camera_start_pose is None:
            self._camera_start_pose = np.array(camera_start_pose).tolist()
           
        if not pixel_ratio is None: 
            self._pixel_ratio = pixel_ratio
            
        if not anti_aliasing is None:     
            self._anti_aliasing = anti_aliasing
            
        if not local_host_port is None: 
            self._local_host_port = local_host_port


        
    def gen_code(self):
        """Generate code for injection."""
        
        #Check ld_urls
        img_types = ['png', 'bmp', 'jpg', 'jpeg']
        if not (self.ldr_urls is None):
            for url in self.ldr_urls:
                error = Utils.is_url_available(Utils.url_modified(url,self.local_host_port), img_types)
                if not (error == "ok!"):
                    raise Exception("The parameter 'url' " + error)

        #Initialize string injection
        _STRJAVASCRIPT = "<html>\n"

        _STRJAVASCRIPT += "<style>\n"
        _STRJAVASCRIPT += ".controller:hover{opacity:1 !important;}\n"
        _STRJAVASCRIPT += "</style>\n"

        _STRJAVASCRIPT += "<body>\n"

        _STRJAVASCRIPT += "<div id='canvas_container_##SIMID##' style='width:##WIDTH##px;height:##HEIGHT##px;position:relative'>\n"
        _STRJAVASCRIPT += "<div id='loading_screen_##SIMID##' style='width:##WIDTH##px;height:##HEIGHT##px;position:relative; " \
                        "background-color: ##LOADSCREENCOLOR##;text-align:center;align-items:center;display:flex;justify-content:center'> \n "
        
        
        _STRJAVASCRIPT += "<img src='https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/SVG/logo_uai_bot.svg' style='width:200px;height:114px'/>\n "
        
        
        _STRJAVASCRIPT += "</div>\n"
        _STRJAVASCRIPT += "<script id='MathJax-script' async src='https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js'></script>\n"
        _STRJAVASCRIPT += "<canvas id='scene_##SIMID##' width='##WIDTH##px' height='##HEIGHT##px'></canvas>\n"
        _STRJAVASCRIPT += "<!-- USER DIVS GO HERE -->"
        _STRJAVASCRIPT += "<div class = 'controller' style='width:##WIDTH##px;height:30px;'></div>\n"
        _STRJAVASCRIPT += "</div>\n"
        _STRJAVASCRIPT += "\n <script type=\"module\">\n"

        p = Path(__file__).with_name('threejs_sim.js')

        for line in open(p.absolute()).readlines():
            _STRJAVASCRIPT += line

        _STRJAVASCRIPT += "\n </script>"
        
        _STRJAVASCRIPT += "<style>\n"
        _STRJAVASCRIPT += "html, body {\n"
        _STRJAVASCRIPT += "margin: 0; \n"
        _STRJAVASCRIPT += "padding: 0; \n"
        _STRJAVASCRIPT += "overflow: hidden; \n"
        _STRJAVASCRIPT += "width: 100%; \n"
        _STRJAVASCRIPT += "height: 100%; \n"
        _STRJAVASCRIPT += "}\n"

        _STRJAVASCRIPT += "canvas {\n"
        _STRJAVASCRIPT += "display: block;\n"
        _STRJAVASCRIPT += "width: 100%; \n"
        _STRJAVASCRIPT += "height: 100%;\n"
        _STRJAVASCRIPT += "}\n"
        _STRJAVASCRIPT += "</style>\n"
              
        _STRJAVASCRIPT += "\n </body>"
        _STRJAVASCRIPT += "\n </html>"
            
        string = _STRJAVASCRIPT
        
        #Make modifications if local host
        if not (self.local_host_port == ''):
            string = re.sub('https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/SVG/logo_uai_bot.svg',
                            'http://localhost:'+self.local_host_port+'/SVG/logo_uai_bot.svg', 
                            string)
                    
            string = re.sub('https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/tex-mml-chtml.js', 
                            string)
            
            string = re.sub('https://cdn.skypack.dev/three@0.135.0/build/three.module.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/build/three.module.js', 
                            string)
            
            string = re.sub('https://cdn.skypack.dev/three@0.135.0/examples/jsm/controls/OrbitControls.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/controls/OrbitControls.js', 
                            string)

            string = re.sub('https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/OBJLoader.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/loaders/OBJLoader.js', 
                            string)
            
            string = re.sub('https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/STLLoader.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/loaders/STLLoader.js', 
                            string)
            
            string = re.sub('https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/MTLLoader.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/loaders/MTLLoader.js', 
                            string)
            
            string = re.sub('https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/GLTFLoader.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/loaders/GLTFLoader.js', 
                            string)

            string = re.sub('https://cdn.skypack.dev/three@0.135.0/examples/jsm/loaders/ColladaLoader.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/loaders/ColladaLoader.js', 
                            string)
            
            string = re.sub('https://cdn.skypack.dev/three@0.135.0/examples/jsm/geometries/ParametricGeometry.js',
                            'http://localhost:'+self.local_host_port+'/Scripts/threejs/geometries/ParametricGeometry.js', 
                            string)
            
            string = re.sub('https://cdn.skypack.dev/dat.gui',
                            'http://localhost:'+self.local_host_port+'/Scripts/dat.gui.module.js', 
                            string)
            
                                                                                                                                
        aa_value = 'true' if self.anti_aliasing else 'false'
        string = string.replace(
            'antialias: true',
            f'antialias: {aa_value}'
        )

        # Replace window.devicePixelRatio
        string = string.replace(
            'window.devicePixelRatio',
            f'window.devicePixelRatio*{self.pixel_ratio}'
        )
        

        for obj in self.list_of_objects:
            if Utils.get_uaibot_type(obj) == "uaibot.HTMLDiv":
                string = re.sub("<!-- USER DIVS GO HERE -->",
                                "<div id='" + obj.name + "'></div>\n <!-- USER DIVS GO HERE -->",
                                string)

        string = re.sub("//SIMULATION PARAMETERS GO HERE", "const delay = 500; \n //SIMULATION PARAMETERS GO HERE",
                        string)

        max_time = 0
        for obj in self.list_of_objects:
            
            code = obj.gen_code(port = self.local_host_port)
            string = re.sub("//USER INPUT GOES HERE", code, string)
            max_time = max(max_time, obj._max_time)

        sim_id = str(time.time()).replace(".","")

        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const maxTime = " + str(max_time) + "; \n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const cameraType= '" + self.camera_type + "'; \n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const sceneID= '" + sim_id + "'; \n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const showWorldFrame="+("true" if self.show_world_frame else "false")+"; \n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const showGrid="+("true" if self.show_grid else "false")+"; \n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const backgroundColor='"+self.background_color+"'; \n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const ambientLightIntensity = " + str(self.ambient_light_intensity) + ";\n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const ldrUrls = " + (
                            str([Utils.url_modified(url,self.local_host_port) for url in self.ldr_urls]) if not 
                            (self.ldr_urls is None) else "[]") + ";\n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("//SIMULATION PARAMETERS GO HERE",
                        "const cameraStartPose = "+str(self.camera_start_pose)+";\n //SIMULATION PARAMETERS GO HERE",
                        string)
        string = re.sub("##WIDTH##", str(self.width), string)
        string = re.sub("##HEIGHT##", str(self.height), string)
        string = re.sub("##HEIGHTLOGO##", str(round(0.57*self.width)), string)
        string = re.sub("##LOADSCREENCOLOR##", self.load_screen_color, string)
        string = re.sub("##SIMID##", sim_id, string)


        return string

    def run(self) -> None:
        """Run simulation."""

        if Utils.get_environment()=='Local':
            
            if self.local_host_port=='':
                rand_suffix = random.randint(10**10, 10**12)
                file_name = f"sim_{rand_suffix}"

                frame = inspect.stack()[1] 
                caller_filepath = frame.filename
                current_folder = os.path.dirname(os.path.abspath(caller_filepath))
                self.save(current_folder, file_name)

                abs_path = Path(current_folder+"/"+file_name+".html").resolve()
                file_url = abs_path.as_uri()
                webbrowser.open(file_url, new=1)
            else:
                print("'run' functionality does not work in localhost. Please \
                      ensure that the html file is saved inside your localhost folder \
                      and open it in your browser through the localhost address. ")

        else:
            display(HTML(self.gen_code()))

    def save(self, address: Optional[str] = None, file_name : str = 'sim') -> None:
        """
    Save the simulation as a self-contained HTML file.

    Parameters
    ----------
    address : string
        The address of the path (example "D:\\"). Default is 'None', and saves
        to the folder that the script was run.
        (default: 'None').
    file_name: string
        The name of the file ("the .html" extension should not appear).
        (default: 'sim')

    """
        if (not (str(type(address)) == "<class 'str'>")) and (not address is None):
            raise Exception(
                "The parameter 'address' should be a string.")
        if not (str(type(file_name)) == "<class 'str'>"):
            raise Exception(
                "The parameter 'file_name' should be a string.")

        try:
            
            if address is None:
                frame = inspect.stack()[1] 
                caller_filepath = frame.filename
                current_folder = os.path.dirname(os.path.abspath(caller_filepath))
            else:
                current_folder = address

            file = open(current_folder + "/" + file_name + ".html", "w+")
            file.write(self.gen_code())
            file.close()
        except:
            raise Exception("Could not open the path '"+current_folder+"' and create the file '"+file_name+".html'.")

    def scan_group(self, group):
        for obj in group.list_of_objects:
            if Utils.get_uaibot_type(obj) == 'uaibot.Robot' and obj.eef_frame_visible:
                self.add(obj._eef_frame)
            if Utils.get_uaibot_type(obj) == 'uaibot.Group':
                self.scan_group(obj)
                
            
    def add(self, obj_sim: List["SimObject"]) -> None:
        """
    Add an object to the simulation. It should be an object that
    can be simulated (Utils.is_a_obj_sim(obj) is true).

    Parameters
    ----------
    obj_sim : object or list of object
        The object(s) to be added to simulation.
    """

        if str(type(obj_sim)) == "<class 'list'>":
            for obj in obj_sim:
                self.add(obj)
        else:
            # Error handling
            if not Utils.is_a_obj_sim(obj_sim):
                raise Exception("The parameter 'obj' should be one of the following: " + str(Utils.IS_OBJ_SIM) + ".")

            if obj_sim.name in self.list_of_names:
                raise Exception("The name '" + obj_sim.name + "' is already in the list of symbols.")

            # end error handling

            self._list_of_names.append(obj_sim.name)
            self._list_of_objects.append(obj_sim)
            
            if Utils.get_uaibot_type(obj_sim) == 'uaibot.Robot' and obj_sim.eef_frame_visible:
                self.add(obj_sim._eef_frame)
            if Utils.get_uaibot_type(obj_sim) == 'uaibot.Group':
                self.scan_group(obj_sim)

    def run_in_browser(self) -> None:
        """Run simulation in browser."""

        if Utils.get_environment()=='Local':
            
            if self.local_host_port=='':
                rand_suffix = random.randint(10**10, 10**12)
                file_name = f"sim_{rand_suffix}"

                frame = inspect.stack()[1] 
                caller_filepath = frame.filename
                current_folder = os.path.dirname(os.path.abspath(caller_filepath))
                self.save(current_folder, file_name)

                abs_path = Path(current_folder+"/"+file_name+".html").resolve()
                file_url = abs_path.as_uri()
                webbrowser.open(file_url, new=1)
            else:
                print("'run' functionality does not work in localhost. Please \
                      ensure that the html file is saved inside your localhost folder \
                      and open it in your browser through the localhost address. ")

        else:
            tmp_dir = tempfile.gettempdir()
            file_name = "uaibot_simulation"
            file_path = os.path.join(tmp_dir, file_name + ".html")
            path = Path(file_path).expanduser().resolve()
            self.save(tmp_dir, file_name)
            # Convert to file:// URL and open
            webbrowser.open_new_tab(path.as_uri())


