from uaibot.utils import *
import numpy as np


def _jac_jac_geo(self, q=None, axis='eef', htm=None):
    if q is None:
        q = self.q
    if htm is None:
        htm = np.matrix(self.htm)

    n = len(self.links)

    # Error handling
    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not (axis == "eef" or axis == "dh" or axis == "com"):
        raise Exception("The parameter 'axis' should be one of the following strings:\n" \
                        "'eef': End-effector \n" \
                        "'dh': All " + str(n) + " axis of Denavit-Hartenberg\n" \
                                                "'com': All " + str(
            n) + " axis centered at the center of mass of the objects.")

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")
    # end error handling

    htm = Utils.cvt(htm)
    if axis == 'com':
        joint_jac, joint_frames = self.jac_geo(q=q, htm=htm, axis='dh', mode='python')
        jac_geo, target_frames = self.jac_geo(q=q, htm=htm, axis='com')
    else:
        joint_jac, joint_frames = self.jac_geo(q=q, htm=htm, axis='dh')
        jac_geo, target_frames = joint_jac, joint_frames

    jj_geo = []

    htm0 = htm * self._htm_base_0

    for i in range(n):
        p_i = target_frames[i][0:3, 3]
        L_i = []
        for j in range(i + 1):

            L_ij = np.matrix(np.zeros((6, n)))

            if j > 0:
                p_j_ant = joint_frames[j - 1][0:3, 3]
                z_j_ant = joint_frames[j - 1][0:3, 2]

                if self.links[j].joint_type == 0:
                    L_ij[0:3, :] = Utils.S(p_i - p_j_ant) * Utils.S(z_j_ant) @ joint_jac[j - 1][3:6, :] + Utils.S(
                        z_j_ant) * (jac_geo[i][0:3, :] - joint_jac[j - 1][0:3, :])
                    L_ij[3:6, :] = -Utils.S(z_j_ant) * joint_jac[j - 1][3:6, :]

                if self.links[j].joint_type == 1:
                    L_ij[0:3, :] = -Utils.S(z_j_ant) * joint_jac[j - 1][3:6, :]
                    # L_ij[3:6, :] = np.zeros((3, n))

            else:
                p_j_ant = htm0[0:3, 3]
                z_j_ant = htm0[0:3, 2]

                if self.links[j].joint_type == 0:
                    L_ij[0:3, :] = Utils.S(z_j_ant) * jac_geo[i][0:3, :]
                    # L_ij[3:6, :] = np.zeros((3, n))

                # if self.links[j].joint_type == 1:
                #    L_ij[0:3, :] = np.zeros((3, n))
                #    L_ij[3:6, :] = np.zeros((3, n))

            L_i.append(L_ij)

        jj_geo.append(L_i)

    if axis == 'dh' or axis == 'com':
        return jj_geo, jac_geo, target_frames

    if axis == 'eef':
        return jj_geo[-1], jac_geo[-1], target_frames[-1]
