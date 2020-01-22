import numpy as np
from math import sin, cos

def Quaternion2Euler(quaternions):
    e0, e1, e2, e3 = quaternions[0], quaternions[1], quaternions[2], quaternions[3]
    phi = np.arctan2(2*(e0*e1 + e2*e3), (e0**2 + e3**2 - e1**2 - e2**2))
    theta = np.arcsin(2*(e0*e2 - e1*e3))
    psi = np.arctan2(2*(e0*e3 + e1*e2), (e0**2 + e1**2 - e2**2 - e3**2))
    return phi, theta, psi

def Euler2Quaternion(phi, theta, psi):
    cosphi2 = cos(phi/2)
    sinphi2 = sin(phi/2)
    costheta2 = cos(theta/2)
    sintheta2 = sin(theta/2)
    cospsi2 = cos(psi/2)
    sinpsi2 = sin(psi/2)
    e0 = cospsi2*costheta2*cosphi2 + sinpsi2*sintheta2*sinphi2
    e1 = cospsi2*costheta2*sinphi2 - sinpsi2*sintheta2*cosphi2
    e2 = cospsi2*sintheta2*cosphi2 + sinpsi2*costheta2*sinphi2
    e3 = sinpsi2*costheta2*cosphi2 - cospsi2*sintheta2*sinphi2
    return e0, e1, e2, e3 