from uaibot.utils import *
import numpy as np


def _dyn_model(self, q, qdot):
    n = len(self.links)

    # Error handling
    if not Utils.is_a_vector(qdot, n):
        raise Exception("The parameter 'qdot' should be a " + str(n) + " dimensional vector.")

    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")
    # end error handling

    jj_geo, jac_geo, htm_com = self.jac_jac_geo(q=q, axis='com')

    i_mat_rot = []

    for i in range(n):
        i_mat = self.links[i].inertia_matrix + self.links[i].mass * Utils.S(self.links[i].com_coordinates) * Utils.S(
            self.links[i].com_coordinates)
        i_mat_rot.append(htm_com[i][0:3, 0:3] * i_mat * htm_com[i][0:3, 0:3].T)

    list_pjac_i_pk = []

    for i in range(n):
        list = []
        for k in range(n):
            pjac_i_pk = np.matrix(np.zeros((6, n)))
            for j in range(i + 1):
                pjac_i_pk[:, j] = jj_geo[i][j][:, k]

            list.append(pjac_i_pk)

        list_pjac_i_pk.append(list)

    list_pm_pk = []

    for k in range(n):
        pm_pk = np.zeros((n, n))
        for i in range(k, n):
            pm_pk += self.links[i].mass * list_pjac_i_pk[i][k][0:3, :].T * jac_geo[i][0:3, :]
            pm_pk += (list_pjac_i_pk[i][k][3:6, :].T + jac_geo[i][3:6, :].T * Utils.S(
                jac_geo[i][3:6, k])) * i_mat_rot[i] * jac_geo[i][3:6, :]

        list_pm_pk.append(pm_pk + (pm_pk).T)

    # Compute the M matrix

    dyn_m = np.zeros((n, n))

    for i in range(n):
        dyn_m += self.links[i].mass * jac_geo[i][0:3, :].T * jac_geo[i][0:3, :]
        dyn_m += jac_geo[i][3:6, :].T * i_mat_rot[i] * jac_geo[i][3:6, :]

    # Compute the C matrix

    dyn_c1 = np.matrix(np.zeros((n, 1)))
    dyn_c2 = np.matrix(np.zeros((n, 1)))
    qdot_v = np.matrix(qdot).reshape((n, 1))

    for j in range(n):
        dyn_c1 = dyn_c1 + (qdot_v[j,0] * list_pm_pk[j]) * qdot_v

    for i in range(n):
        dyn_c2[i] = 0.5 * qdot_v.T * list_pm_pk[i] * qdot_v

    dyn_c = (dyn_c1 - dyn_c2).reshape((n, 1))

    # Compute the G matrix

    dyn_g = np.zeros((n, 1))
    GRAVITY_ACC = 9.8
    for i in range(n):
        dyn_g += GRAVITY_ACC * self.links[i].mass * jac_geo[i][2, :].reshape((n, 1))

    return dyn_m, dyn_c, dyn_g
