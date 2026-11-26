from ._add_ani_frame import _record_ani_frame


def _set_ani_frame(self, q=None, htm=None, enforce_joint_limits=False):
    q, pose = self._resolve_pose(self.q0 if q is None else q, htm, enforce_joint_limits)
    self._set_pose_and_state(q, pose, 0.0, reset=True)
    self._frames = []
    self.code = ""
    self._max_time = 0.0
    _record_ani_frame(self, 0.0, reset=True)
