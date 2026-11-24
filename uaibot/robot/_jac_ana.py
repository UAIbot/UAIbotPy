from uaibot.utils import *
import numpy as np


# Analytic Jacobian
def _jac_ana(self, q=None, htm=None):
    if q is None:
        q = self.q
    if htm is None:
        htm = np.matrix(self.htm)

    n = len(self.links)

    # Error handling
    if not Utils.is_a_vector(q, n):
        raise Exception("The parameter 'q' should be a " + str(n) + " dimensional vector.")

    if not Utils.is_a_matrix(htm, 4, 4):
        raise Exception("The parameter 'htm' should be a 4x4 homogeneous transformation matrix.")
    # end error handling

    jac_geo, htm_eef = self.jac_geo(Utils.cvt(q),'eef',htm)

    alpha, beta, gamma = Utils.euler_angles(htm_eef)

    sa = np.sin(alpha)
    ca = np.cos(alpha)
    cb = np.cos(beta)
    tb = np.tan(beta)

    transf = np.matrix([[-ca*tb, -sa*tb, 1],
                        [-sa   ,     ca, 0],
                        [ ca/cb,  sa/cb, 0]])

    jac_ana = np.matrix(np.zeros((6,n)))
    jac_ana[0:3,:] = jac_geo[0:3,:]
    jac_ana[3:6,:] = transf * jac_geo[3:6,:]

    phi = np.matrix([[alpha],[beta],[gamma]])

    return jac_ana, htm_eef, phi
