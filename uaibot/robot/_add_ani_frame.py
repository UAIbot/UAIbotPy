"""Update a manipulator's native truth and write the matching scene frame."""

import numpy as np

from ._dynamics import frame_time


def _add_ani_frame(self, time, q=None, htm=None, enforce_joint_limits=False):
    time = frame_time(time)
    q, pose = self._resolve_pose(q, htm, enforce_joint_limits)
    self._set_pose_and_state(q, pose, time)
    _record_ani_frame(self, time)


def _record_ani_frame(self, time, *, reset=False):
    pose = np.asarray(self._htm).ravel().tolist()
    self._frames.append([time, *pose, np.asarray(self._q).ravel().tolist()])
    self._max_time = max(self._max_time, time)

    def record(obj, pose):
        if reset:
            obj.set_ani_frame(htm=pose)
        else:
            obj.add_ani_frame(time, htm=pose)

    if self.eef_frame_visible or self.attached_objects:
        tool_pose = self.fkm()
        if self.eef_frame_visible:
            record(self._eef_frame, tool_pose)
        for obj, local_pose in self.attached_objects:
            record(obj, tool_pose @ local_pose)

    collision_objects = [link.col_objects for link in self._links]
    if any(collision_objects):
        for link_pose, objects in zip(self.fkm(axis="dh"), collision_objects):
            for obj, local_pose in objects:
                record(obj, link_pose @ local_pose)
