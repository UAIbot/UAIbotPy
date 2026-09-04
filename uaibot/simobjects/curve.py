import numpy as np
from uaibot.utils import Utils
from uaibot.simobjects import Frame, Group


class Curve:
    """A dynamic point cloud that can change its shape over time by
    switching between discrete keyframes.

    Each keyframe stores a complete set of 3D points. When the
    simulation time reaches a keyframe's timestamp, the displayed points
    are updated to that set. No interpolation is performed between
    keyframes; the change is instantaneous. If smooth transitions are
    desired, users should provide a sufficient number of intermediate
    keyframes.

    Parameters
    ----------
    name : string
        The object's name. (default: '' (automatic)).
    size : positive float
        The size of each point in the curve.
    color : string
        A HTML-compatible color.
    points : a Nx3 numpy array or list of 3D points
        Initial point positions (used at time 0).
    """

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    @property
    def name(self):
        """The object name."""
        return self._name

    @property
    def size(self):
        """The size of each point in the curve."""
        return self._size

    @property
    def color(self):
        """Color of the object, a HTML-compatible string."""
        return self._color

    @property
    def points(self):
        """The current points of the curve."""
        return np.array(self._points)

    @property
    def inital_points(self):
        """The initial points of the curve."""
        return np.array(self._initial_points)

    @property
    def cpp_obj(self):
        """Used in the c++ interface (not implemented for Curve)."""
        return self._cpp_obj

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_points(points):
        """
        Converts input points to a Nx3 numpy array.
        Accepts:
            - numpy matrix (Nx3)
            - numpy array (3, N) or (N, 3)
            - list of lists/arrays of length 3 (each point)
        Returns a numpy array of shape (N, 3).
        """
        if isinstance(points, np.matrix):
            if points.shape[1] != 3:
                raise Exception("Points matrix must have 3 columns.")
            return np.array(points)
        elif isinstance(points, np.ndarray):
            if points.ndim != 2 or points.shape[1] != 3:
                # Try transposing if shape is (3,N)
                if points.ndim == 2 and points.shape[0] == 3:
                    points = points.T
                else:
                    raise Exception("Points array must have shape (3, N) or (N, 3).")
            return points
        elif isinstance(points, list):
            # List of points, each point is list/array of length 3
            if len(points) == 0:
                raise Exception("Points list cannot be empty.")
            point_list = []
            for p in points:
                if isinstance(p, (list, np.ndarray)):
                    p = np.array(p).flatten()
                    if len(p) != 3:
                        raise Exception("Each point must have 3 coordinates.")
                    point_list.append(p)
                else:
                    raise Exception("Invalid point format in list.")
            return np.array(point_list)  # shape (N, 3)
        else:
            raise Exception("Points must be a matrix, ndarray, or list of points.")

    def __init__(self, name="", points=[], size=0.1, color="blue"):
        # Error handling
        if not Utils.is_a_number(size) or size < 0:
            raise Exception("The parameter 'size' should be a positive float")

        if name == "":
            name = "var_curve_id_" + str(id(self))

        if not Utils.is_a_name(name):
            raise Exception(
                "The parameter 'name' should be a string. Only "
                "characters 'a-z', 'A-Z', '0-9' and '_' are allowed. "
                "It should not begin with a number."
            )

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a color")

        # Convert and store initial points
        try:
            self._points = self._convert_points(points)
            self._initial_points = self._points.copy()
        except Exception as e:
            raise Exception(f"Invalid points: {e}")

        # Store number of points (must be consistent across frames)
        self._num_points = self._points.shape[0]

        self._name = name
        self._size = size
        self._color = color
        self._frames = []  # List of [time, points_matrix]
        self._max_time = 0

        # cpp_obj not implemented; keep empty
        self._cpp_obj = []

        # Add initial frame
        self.add_ani_frame(0, self._initial_points)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self):
        string = f"Curve '{self._name}' with {self._num_points} points.\n"
        string += f" Size: {self._size}\n"
        string += f" Color: {self._color}\n"
        string += f" Frames: {len(self._frames)} (last at t={self._max_time:.2f}s)"
        return string

    def __iter__(self):
        """Iterate over the current points (each point is a 1D numpy
        array of shape (3,)).
        """
        for i in range(self._num_points):
            yield self._points[i]

    def __len__(self):
        """Return the number of points in the curve."""
        return self._num_points

    def __getitem__(self, index):
        """Return point(s) from the current frame. Supports integer or
        slice indexing.
        """
        return self._points[index]

    def __copy__(self):
        new_curve = Curve(
            self.name + "_copy", self._initial_points, self.size, self.color
        )
        # Copy animation frames (excluding the initial frame at time 0 added by constructor)
        for time, pts in self._frames[1:]:
            new_curve.add_ani_frame(time, pts)
        return new_curve

    def __deepcopy__(self, memo):
        return self.__copy__()

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def add_ani_frame(self, time, points):
        """Add a new keyframe at the given time with a new set of points.
        The number of points must match the initial point count.

        Parameters
        ----------
        time : float
            Time (in seconds) at which this keyframe should be active.
        points : Nx3 array or list of 3D points
            The point positions at this keyframe.
        """
        if not Utils.is_a_number(time) or time < 0:
            raise Exception("The parameter 'time' should be a nonnegative float.")

        # Convert points and check dimension
        try:
            pts = self._convert_points(points)
        except Exception as e:
            raise Exception(f"Invalid points for animation frame: {e}")

        if pts.shape[0] != self._num_points:
            raise Exception(
                f"All animation frames must have the same number of "
                f"points (expected {self._num_points}, got {pts.shape[0]})."
            )

        time_f = float(time)
        self._points = pts
        self._frames.append([time_f, pts])
        self._max_time = max(self._max_time, time_f)

    def set_ani_frame(self, points):
        """Clear all existing animation frames and set a single keyframe
        at time 0 with the given points. This effectively resets the
        animation to a static curve.

        Parameters
        ----------
        points : Nx3 array or list of 3D points
            The point positions for the new single frame.
        """
        # Validate inputs
        pts = self._convert_points(points)
        # if pts.shape[0] != self._num_points:
        #     raise Exception("The number of points must match the initial points.")

        # Reset frames
        self._frames = []
        self._points = pts
        self._initial_points = pts
        self._num_points = pts.shape[0]
        self.add_ani_frame(0.0, pts)
        self._max_time = 0  # Only one frame at time 0

    def get_points_at_time(self, time):
        """Return the points (Nx3 numpy array) that would be active at
        the given time. If the time is before the first frame, returns
        the first frame's points. If after the last frame, returns the
        last frame's points.
        """
        if not self._frames:
            return self._points  # fallback
        # Find the last frame with time <= requested time
        active_frame = self._frames[0][1]  # default to first
        for t, pts in self._frames:
            if t <= time:
                active_frame = pts
            else:
                break
        return np.array(active_frame)

    def gen_code(self, port):
        """Generate JavaScript code for injection into the simulator."""
        # Convert frames to JS array format: [[time, flatPointsArray], ...]
        js_frames = []
        for time, pts in self._frames:
            flat = [float(coord) for coord in pts.flatten()]
            # flat = []
            # for i in range(pts.shape[0]):
            #     flat.extend([pts[i, 0], pts[i, 1], pts[i, 2]])
            js_frames.append([time, flat])

        string = "\n"
        string += f"//BEGIN DECLARATION OF THE CURVE '{self.name}'\n\n"
        string += f"const var_{self.name} = new Curve({js_frames}, '{self.color}', {self.size});\n"
        string += f"sceneElements.push(var_{self.name});\n"
        string += "//USER INPUT GOES HERE"
        return string

    def transform(self, htm, time=None):
        """
        Apply a homogeneous transformation matrix (4x4) to the
        current points.
        If `time` is provided, the transformed points are added as a
        new animation frame at that time. Otherwise, the current points
        are updated in place and the initial frame is updated as well.
        """
        if htm.shape != (4, 4):
            raise Exception("Transformation must be a 4x4 homogeneous matrix.")

        # Convert to homogeneous coordinates (Nx4)
        ones = np.ones((self._num_points, 1))
        pts_h = np.hstack((self._points, ones))
        transformed = (htm @ pts_h.T).T[:, :3]  # shape (N,3)

        if time is not None:
            self.add_ani_frame(time, transformed)
        else:
            self._points = transformed
            # Also update the first frame if it exists at time 0
            for i, (t, _) in enumerate(self._frames):
                if t == 0:
                    self._frames[i][1] = transformed
                    break
            # Update initial points as well
            self._initial_points = transformed.copy()
        return self


class CurveSE3:
    """A curve in SE(3) that can change its configuration over time by
    switching between discrete keyframes.

    Each keyframe stores a complete set of homogeneous transformation
    matrices (N x 4 x 4), representing the poses of N points along the
    curve. When the simulation time reaches a keyframe's timestamp,
    the displayed poses are updated to that set. No interpolation is
    performed; the change is instantaneous. Users should provide
    sufficient keyframes if smooth transitions are desired.

    Parameters
    ----------
    name : string
        The object's name. (default: '' (automatic)).
    size : positive float
        The size of each point in the curve.
    color : string
        A HTML-compatible color for the points.
    points : N x 4 x 4 numpy array or list of 4x4 matrices
        Initial poses (homogeneous transformation matrices) used at time 0.
    frame_size : positive float
        Size of the orientation frames attached to each point.
    frame_axis_colors : list of 3 HTML-compatible strings
        Colors for the x, y, z axes of the orientation frames.
    frame_axis_names : list of 3 strings
        Names for the x, y, z axes of the orientation frames.
    num_frames : int
        Number of orientation frames to display (uniformly distributed).
        Default is 10.
    frame_indices : list
        List of the point indices for which an orientation frame should
        be displayed. This is ignored if `num_frames` is passed.
    """

    def __init__(
        self,
        name="",
        points=[],
        size=0.1,
        color="blue",
        frame_size=0.3,
        frame_axis_colors=["red", "lime", "blue"],
        frame_axis_names=["x", "y", "z"],
        num_frames=None,
        frame_indices=None,
    ):
        # Error handling
        if not Utils.is_a_number(size) or size < 0:
            raise Exception("The parameter 'size' should be a positive float")

        if not Utils.is_a_number(frame_size) or frame_size < 0:
            raise Exception("The parameter 'frame_size' should be a positive float")

        if name == "":
            name = "var_curvese3_id_" + str(id(self))

        if not Utils.is_a_name(name):
            raise Exception(
                "The parameter 'name' should be a string. Only characters 'a-z', 'A-Z', '0-9' and '_' are allowed. It should not begin with a number."
            )

        if not Utils.is_a_color(color):
            raise Exception("The parameter 'color' should be a color")

        if not isinstance(frame_axis_colors, list) or len(frame_axis_colors) != 3:
            raise Exception(
                "The parameter 'frame_axis_colors' should be a list of 3 colors."
            )
        for c in frame_axis_colors:
            if not Utils.is_a_color(c):
                raise Exception(
                    "Each axis color must be a valid HTML-compatible color."
                )

        if not isinstance(frame_axis_names, list) or len(frame_axis_names) != 3:
            raise Exception(
                "The parameter 'frame_axis_names' should be a list of 3 strings."
            )
        for n in frame_axis_names:
            if not isinstance(n, str):
                raise Exception("Each axis name must be a string.")

        # Convert and store initial HTM
        try:
            self._points = self._convert_points(points)
            self._initial_points = self._points.copy()
        except Exception as e:
            raise Exception(f"Invalid HTM: {e}")

        self._num_poses = self._points.shape[0]

        self._name = name
        self._size = size
        self._color = color
        self._frame_size = frame_size
        self._frame_axis_colors = frame_axis_colors
        self._frame_axis_names = frame_axis_names
        if num_frames is None:
            if frame_indices is None:
                # Defaults to 10 frames
                self._num_frames = 10
                self._frame_indices = np.linspace(
                    0, self._num_poses - 1, 10, dtype=int
                ).tolist()
            else:
                self._frame_indices = frame_indices
                self._num_frames = len(frame_indices)
        else:
            self._num_frames = num_frames
            self._frame_indices = np.linspace(
                0, self._num_poses - 1, num_frames, dtype=int
            ).tolist()
        self._frames = []  # List of [time, htm_array]
        self._max_time = 0
        self._cpp_obj = []

        # Add initial frame
        self.add_ani_frame(0, self._initial_points)

    # ------------------------------------------------------------------
    # Properties for frame display control
    # ------------------------------------------------------------------
    @property
    def frame_indices(self):
        """List of indices of poses where orientation frames are shown."""
        return self._frame_indices

    @frame_indices.setter
    def frame_indices(self, value):
        if isinstance(value, (list, np.ndarray)):
            indices = np.array(value, dtype=int)
            if indices.ndim != 1 or len(indices) == 0:
                raise Exception("frame_indices must be a non-empty 1D list/array.")
            if np.min(indices) < 0 or np.max(indices) >= self._num_poses:
                raise Exception("frame_indices out of range.")
            if len(np.unique(indices)) != len(indices):
                raise Exception("frame_indices must contain unique values.")
            self._frame_indices = sorted(
                indices.tolist()
            )  # keep sorted for consistency
            self._num_frames = len(self._frame_indices)
        else:
            raise Exception("frame_indices must be a list or numpy array.")

    @property
    def num_frames(self):
        """Number of orientation frames to display (uniformly distributed)."""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, value):
        if (
            not isinstance(value, (int, np.integer))
            or value < 1
            or value > self._num_poses
        ):
            raise Exception(
                f"num_frames must be an integer between 1 and {self._num_poses}."
            )
        # Compute uniform indices
        if value == 1:
            indices = [0]  # first pose
        else:
            indices = np.linspace(0, self._num_poses - 1, value, dtype=int).tolist()
        self._frame_indices = indices
        self._num_frames = value

    # ------------------------------------------------------------------
    # Other properties (points, initial_points, translations, etc.)
    # ------------------------------------------------------------------

    @property
    def name(self):
        """The object name."""
        return self._name

    @property
    def size(self):
        """The size of each point in the curve."""
        return self._size

    @property
    def color(self):
        """Color of the points, a HTML-compatible string."""
        return self._color

    @property
    def frame_size(self):
        """Size of the orientation frames."""
        return self._frame_size

    @property
    def frame_axis_colors(self):
        """List of colors for the frame axes."""
        return self._frame_axis_colors

    @property
    def frame_axis_names(self):
        """List of names for the frame axes."""
        return self._frame_axis_names

    @property
    def htm(self):
        """The current poses as a Nx4x4 numpy array."""
        return np.array(self._points)

    @property
    def initial_htm(self):
        """The initial poses as a Nx4x4 numpy array."""
        return np.array(self._initial_points)

    @property
    def positions(self):
        """The current positions (Nx3 array)."""
        return self._points[:, :3, 3]

    @property
    def orientations(self):
        """The current rotation matrices (Nx3x3 array)."""
        return self._points[:, :3, :3]

    @property
    def cpp_obj(self):
        """Used in the c++ interface (not implemented for CurveSE3)."""
        return self._cpp_obj

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self):
        string = f"CurveSE3 '{self._name}' with {self._num_poses} poses.\n"
        string += f" Point size: {self._size}\n"
        string += f" Point color: {self._color}\n"
        string += f" Frame size: {self._frame_size}\n"
        string += f" Frames: {len(self._frames)} (last at t={self._max_time:.2f}s)"
        return string

    def __iter__(self):
        """Iterate over the current poses (each pose is a 4x4 numpy array)."""
        for i in range(self._num_poses):
            yield self._points[i]

    def __len__(self):
        """Return the number of poses in the curve."""
        return self._num_poses

    def __getitem__(self, index):
        """Return pose(s) from the current frame. Supports integer or
        slice indexing.
        """
        return self._points[index]

    def __copy__(self):
        new_curve = CurveSE3(
            name=self.name + "_copy",
            points=self._initial_points,
            size=self.size,
            color=self.color,
            frame_size=self.frame_size,
            frame_axis_colors=self.frame_axis_colors,
            frame_axis_names=self.frame_axis_names,
            frame_indices=self._frame_indices,
        )
        for time, htm in self._frames[1:]:
            new_curve.add_ani_frame(time, htm)
        return new_curve

    def __deepcopy__(self, memo):
        return self.__copy__()

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    @staticmethod
    def _convert_points(points):
        """
        Converts input homogeneous matrices to a Nx4x4 numpy array.
        Accepts:
            - numpy array of shape (N,4,4)
            - list of 4x4 numpy arrays or lists
        Returns a numpy array of shape (N,4,4).
        """
        if isinstance(points, np.ndarray):
            if points.ndim == 3 and points.shape[1:] == (4, 4):
                return points
            else:
                raise Exception("HTM array must have shape (N,4,4).")
        elif isinstance(points, list):
            if len(points) == 0:
                raise Exception("HTM list cannot be empty.")
            htm_list = []
            for h in points:
                h = np.array(h)
                if h.shape != (4, 4):
                    raise Exception("Each element in the list must be a 4x4 matrix.")
                htm_list.append(h)
            return np.array(htm_list)
        else:
            raise Exception(
                "HTM must be a numpy array of shape (N,4,4) or a list of 4x4 matrices."
            )

    def add_ani_frame(self, time, points):
        """
        Add a new keyframe at the given time with a new set of poses.
        The number of poses must match the initial pose count.

        Parameters
        ----------
        time : float
            Time (in seconds) at which this keyframe should be active.
        points : Nx4x4 numpy array or list of 4x4 matrices
            The poses at this keyframe.
        """
        if not Utils.is_a_number(time) or time < 0:
            raise Exception("The parameter 'time' should be a nonnegative float.")

        try:
            points_arr = self._convert_points(points)
        except Exception as e:
            raise Exception(f"Invalid points for animation frame: {e}")

        if points_arr.shape[0] != self._num_poses:
            raise Exception(
                f"All animation frames must have the same number of poses "
                f"(expected {self._num_poses}, got {points_arr.shape[0]})."
            )

        time_f = float(time)
        self._points = points_arr
        self._frames.append([time_f, points_arr])
        self._max_time = max(self._max_time, time_f)

    def set_ani_frame(self, points):
        """
        Clear all existing animation frames and set a single keyframe at time 0
        with the given poses. This effectively resets the animation to a static curve.

        Parameters
        ----------
        points : Nx4x4 numpy array or list of 4x4 matrices
            The poses for the new single frame.
        """
        htm_arr = self._convert_points(points)
        self._frames = []
        self._points = htm_arr
        self._initial_points = htm_arr
        self._num_poses = htm_arr.shape[0]
        self.add_ani_frame(0.0, htm_arr)
        self._max_time = 0

    def get_points_at_time(self, time):
        """Return the poses (Nx4x4 numpy array) that would be active at the given time.
        If the time is before the first frame, returns the first frame's poses.
        If after the last frame, returns the last frame's poses.
        """
        if not self._frames:
            return self._points
        active_frame = self._frames[0][1]
        for t, htm in self._frames:
            if t <= time:
                active_frame = htm
            else:
                break
        return np.array(active_frame)

    def gen_code(self, port):
        """
        Generate JavaScript code by composing a Curve (for positions) and
        multiple Frame objects (for orientations) into a Group.
        """
        # Create a Curve for the positions (translations)
        point_curve = Curve(
            name=self.name + "_points",
            points=self._points[:, :3, 3],  # Nx3 translations
            size=self.size,
            color=self.color,
        )
        # Add animation frames for the positions
        for time, pts in self._frames[1:]:  # skip initial frame (already added)
            point_curve.add_ani_frame(time, pts[:, :3, 3])

        curve_code = point_curve.gen_code(port)

        # For each pose index, create a Frame that follows that pose over time
        frame_objects = []
        frame_codes = []
        for i in self._frame_indices:
            # Initial pose for this index
            initial_pose = self._initial_points[i]
            fr = Frame(
                name=f"{self.name}_frame_{i}",
                htm=initial_pose,
                size=self.frame_size,
                axis_color=self.frame_axis_colors,
                axis_names=self.frame_axis_names,
            )
            # Add animation frames for this pose index from all keyframes
            for time, pts in self._frames[1:]:
                fr.add_ani_frame(time, pts[i])
            frame_objects.append(fr)
            frame_codes.append(fr.gen_code(port))

        # !!!!
        # TODO: Currently the Group class does not work as expected.
        # It relies on htm multiplication, which are absent in Curve
        # This should be corrected in future versions
        # !!!!
        # Create a group containing the point curve and all frames
        # group = Group(name=self.name, list_of_objects=[point_curve] + frame_objects)
        # Return the generated code from the group
        # return group.gen_code(port=port)

        # Current workaround is generating the block manually:
        # 3. Combine code strings, removing "//USER INPUT GOES HERE" from all but the last
        combined = curve_code
        for fc in frame_codes:
            combined += "\n" + fc.replace("//USER INPUT GOES HERE", "")
        # Ensure final marker is present
        if "//USER INPUT GOES HERE" not in combined:
            combined += "\n//USER INPUT GOES HERE"
        return combined
