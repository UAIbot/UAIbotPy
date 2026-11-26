from uaibot.utils import *
import numpy as np
import os


# Geometric Jacobian
def _jac_geo(self, q, axis, htm, mode):
    if q is None:
        q = self.q
    if htm is None:
        htm = np.matrix(self.htm)

    n = len(self.links)

    # Error handling
    if mode not in ["python", "c++", "auto"]:
        raise Exception("The parameter 'mode' should be 'python,'c++', or 'auto'.")

    if not Utils.is_a_vector(q, n):
        raise Exception(
            "The parameter 'q' should be a " + str(n) + " dimensional vector."
        )

    if not (axis == "eef" or axis == "dh" or axis == "com"):
        raise Exception(
            "The parameter 'axis' should be one of the following strings:\n"
            "'eef': End-effector \n"
            "'dh': All " + str(n) + " axis of Denavit-Hartenberg\n"
            "'com': All "
            + str(n)
            + " axis centered at the center of mass of the objects."
        )

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception(
            "The parameter 'htm' should be a 4x4 homogeneous transformation matrix."
        )
    if mode == "c++" and os.environ["CPP_SO_FOUND"] == "0":
        raise Exception("c++ mode is set, but .so file was not loaded!")
    # end error handling

    if (
        mode == "python"
        or axis == "com"
        or (mode == "auto" and os.environ["CPP_SO_FOUND"] == "0")
    ):
        return _jac_geo_python(self, Utils.cvt(q), axis, Utils.cvt(htm))
    else:
        fk_res = self.cpp_robot.fk(Utils.cvt(q), htm, True)

        if axis == "eef":
            return Utils.cvt(np.vstack((fk_res.jac_v_ee, fk_res.jac_w_ee))), Utils.cvt(
                fk_res.htm_ee
            )
        else:
            htm_dh = []
            jac_dh = []
            for i in range(n):
                htm_dh.append(Utils.cvt(fk_res.htm_dh[i]))
                jac_dh.append(
                    Utils.cvt(np.vstack((fk_res.jac_v_dh[i], fk_res.jac_w_dh[i])))
                )

            return jac_dh, htm_dh


def _jac_geo_python(self, q=None, axis="eef", htm=None):

    n = len(self.links)

    htm_dh = self.fkm(q, "dh", htm, mode="python")
    htm_for_jac = htm_dh if axis != "com" else self.fkm(q, "com", htm, mode="python")

    jac = [np.matrix(np.zeros((6, n))) for i in range(n)]

    htm_world_0 = htm * self.htm_base_0

    for i in range(n):
        p_i = htm_for_jac[i][0:3, 3]
        for j in range(i + 1):

            if j == 0:
                p_j_ant = htm_world_0[0:3, 3]
                z_j_ant = htm_world_0[0:3, 2]
            else:
                p_j_ant = htm_dh[j - 1][0:3, 3]
                z_j_ant = htm_dh[j - 1][0:3, 2]

            if self.links[j].joint_type == 0:
                jac[i][0:3, j] = Utils.S(z_j_ant) * (p_i - p_j_ant)
                jac[i][3:6, j] = z_j_ant

            if self.links[j].joint_type == 1:
                jac[i][0:3, j] = z_j_ant
                jac[i][3:6, j] = np.matrix(np.zeros((3, 1)))

    if axis == "dh" or axis == "com":
        return jac, htm_for_jac

    if axis == "eef":
        htm_0_eef = htm_for_jac[-1][:, :] * self.htm_n_eef
        jg = jac[-1][:, :]
        jv = jg[0:3, :]
        jw = jg[3:6, :]
        jv = jv + Utils.S(htm_for_jac[-1][0:3, -1] - htm_0_eef[0:3, -1]) * jw
        jg = np.block([[jv], [jw]])

        return jg, htm_0_eef
