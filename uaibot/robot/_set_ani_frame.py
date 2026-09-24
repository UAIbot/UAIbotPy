from uaibot.utils import *


# Set config. Restart animation queue
def _set_ani_frame(self, q=None, htm=None, enforce_joint_limits=False):
    if q is None:
        q = self.q0

    if htm is None:
        htm = self.htm

    # Error handling
    if not Utils.is_a_vector(q, len(self.links)):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")

    if not str(type(enforce_joint_limits)) == "<class 'bool'>":
        raise Exception("The parameter 'enforce_joint_limits' must be a boolean.")
    # end error handling

    self._frames = []
    self.code = ''
    self._max_time = 0
    self.add_ani_frame(0, q, htm, enforce_joint_limits)
    
    if self.eef_frame_visible:
        self._eef_frame.add_ani_frame(0,self.fkm(q))
        
    self.update_col_object(0)
