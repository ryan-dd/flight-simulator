import numpy as np
from math import sin, cos
from scipy.spatial.transform import Rotation


def Quaternion2Euler(quaternions):
    e0, e1, e2, e3 = quaternions[0], quaternions[1], quaternions[2], quaternions[3]
    phi = np.arctan2(2*(e0*e1 + e2*e3), (e0**2 + e3**2 - e1**2 - e2**2))
    theta = np.arcsin(2*(e0*e2 - e1*e3))
    psi = np.arctan2(2*(e0*e3 + e1*e2), (e0**2 + e1**2 - e2**2 - e3**2))
    return phi.item(0), theta.item(0), psi.item(0)

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

def Euler2Rotation(phi, theta, psi):
    """
    Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
    """
    # only call sin and cos once for each angle to speed up rendering
    c_phi = np.cos(phi).item()
    s_phi = np.sin(phi).item()
    c_theta = np.cos(theta).item()
    s_theta = np.sin(theta).item()
    c_psi = np.cos(psi).item()
    s_psi = np.sin(psi).item()

    R_roll = np.array([[1, 0, 0],
                        [0, c_phi, s_phi],
                        [0, -s_phi, c_phi]])
    R_pitch = np.array([[c_theta, 0, -s_theta],
                        [0, 1, 0],
                        [s_theta, 0, c_theta]])
    R_yaw = np.array([[c_psi, s_psi, 0],
                        [-s_psi, c_psi, 0],
                        [0, 0, 1]])
    R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 in book)
    return R.T  # transpose to return body to inertial

def Quaternion2Rotation(e):
    e0 = e.item(0)
    e1 = e.item(1)
    e2 = e.item(2)
    e3 = e.item(3)

    R = np.array([
        [e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)],
        [2 * (e1 * e2 + e0 * e3), e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2, 2 * (e2 * e3 - e0 * e1)],
        [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2]
                ])
    return R

if __name__ == "__main__":
    # Test the subjects and compare to scipy
    phi, theta, psi = Quaternion2Euler([0.999, 0.02, 0.03, 0.04])
    e0, e1, e2, e3 = Euler2Quaternion(phi, theta, psi)
    Rot = Euler2Rotation(phi, theta, psi)
    result = Rotation.from_euler('xyz', [phi, theta, psi])
    result.as_dcm
    result.as_euler('xyz')
    hi=0