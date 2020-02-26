"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion, Quaternion2Euler
from tools.transfer_function import TransferFunction
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts

def compute_model(mav, trim_state, trim_input):
    # A_lon, B_lon, A_lat, B_lat = compute_ss_model(mav, trim_state, trim_input)
    Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, \
    a_V1, a_V2, a_V3 = compute_tf_model(mav, trim_state, trim_input)

    # write transfer function gains to file
    file = open('model_coef.py', 'w')
    file.write('import numpy as np\n')
    file.write('x_trim = np.array([[%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f]]).T\n' %
               (trim_state.item(0), trim_state.item(1), trim_state.item(2), trim_state.item(3),
                trim_state.item(4), trim_state.item(5), trim_state.item(6), trim_state.item(7),
                trim_state.item(8), trim_state.item(9), trim_state.item(10), trim_state.item(11),
                trim_state.item(12)))
    file.write('u_trim = np.array([[%f, %f, %f, %f]]).T\n' %
               (trim_input.item(0), trim_input.item(1), trim_input.item(2), trim_input.item(3)))
    file.write('Va_trim = %f\n' % Va_trim)
    file.write('alpha_trim = %f\n' % alpha_trim)
    file.write('theta_trim = %f\n' % theta_trim)
    file.write('a_phi1 = %f\n' % a_phi1)
    file.write('a_phi2 = %f\n' % a_phi2)
    file.write('a_theta1 = %f\n' % a_theta1)
    file.write('a_theta2 = %f\n' % a_theta2)
    file.write('a_theta3 = %f\n' % a_theta3)
    file.write('a_V1 = %f\n' % a_V1)
    file.write('a_V2 = %f\n' % a_V2)
    file.write('a_V3 = %f\n' % a_V3)
    # file.write('A_lon = np.array([\n    [%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f]])\n' %
    # (A_lon[0][0], A_lon[0][1], A_lon[0][2], A_lon[0][3], A_lon[0][4],
    #  A_lon[1][0], A_lon[1][1], A_lon[1][2], A_lon[1][3], A_lon[1][4],
    #  A_lon[2][0], A_lon[2][1], A_lon[2][2], A_lon[2][3], A_lon[2][4],
    #  A_lon[3][0], A_lon[3][1], A_lon[3][2], A_lon[3][3], A_lon[3][4],
    #  A_lon[4][0], A_lon[4][1], A_lon[4][2], A_lon[4][3], A_lon[4][4]))
    # file.write('B_lon = np.array([\n    [%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f]])\n' %
    # (B_lon[0][0], B_lon[0][1],
    #  B_lon[1][0], B_lon[1][1],
    #  B_lon[2][0], B_lon[2][1],
    #  B_lon[3][0], B_lon[3][1],
    #  B_lon[4][0], B_lon[4][1],))
    # file.write('A_lat = np.array([\n    [%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f],\n    '
    #            '[%f, %f, %f, %f, %f]])\n' %
    # (A_lat[0][0], A_lat[0][1], A_lat[0][2], A_lat[0][3], A_lat[0][4],
    #  A_lat[1][0], A_lat[1][1], A_lat[1][2], A_lat[1][3], A_lat[1][4],
    #  A_lat[2][0], A_lat[2][1], A_lat[2][2], A_lat[2][3], A_lat[2][4],
    #  A_lat[3][0], A_lat[3][1], A_lat[3][2], A_lat[3][3], A_lat[3][4],
    #  A_lat[4][0], A_lat[4][1], A_lat[4][2], A_lat[4][3], A_lat[4][4]))
    # file.write('B_lat = np.array([\n    [%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f],\n    '
    #            '[%f, %f]])\n' %
    # (B_lat[0][0], B_lat[0][1],
    #  B_lat[1][0], B_lat[1][1],
    #  B_lat[2][0], B_lat[2][1],
    #  B_lat[3][0], B_lat[3][1],
    #  B_lat[4][0], B_lat[4][1],))
    file.write('Ts = %f\n' % Ts)
    file.close()


def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    mav._state = trim_state
    mav._update_velocity_data()
    Va_trim = mav._Va
    alpha_trim = mav._alpha
    phi, theta_trim, psi = Quaternion2Euler(trim_state[6:10])

    Va_star = trim_state[3]
    alpha_star = MAV.alpha0
    delta_e_star = trim_input[0].item(0)
    delta_t_star = trim_input[1].item(0) 

    # define transfer function constants
    a_phi1 = -1/2*MAV.rho*mav._Va**2 * MAV.S_wing*MAV.b*MAV.C_p_p * MAV.b/(2*mav._Va)
    a_phi2 = 1/2*MAV.rho*mav._Va**2 * MAV.S_wing*MAV.b*MAV.C_p_delta_a
    a_theta1 = -MAV.rho*mav._Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_q * MAV.c/(2*mav._Va)
    a_theta2 = -MAV.rho*mav._Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_alpha
    a_theta3 = MAV.rho*mav._Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_delta_e
    a_V1 = MAV.rho*Va_star*MAV.S_wing/MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha*alpha_star + MAV.C_D_delta_e * delta_e_star) + MAV.rho * MAV.S_prop/MAV.mass * MAV.C_prop*Va_star
    a_V1 = a_V1.item(0)
    a_V2 = MAV.rho * MAV.S_prop/MAV.mass * MAV.C_prop*MAV.k_motor**2*delta_t_star
    a_V3 = MAV.gravity

    return Va_trim, alpha_trim, theta_trim, a_phi1, a_phi2, a_theta1, a_theta2, a_theta3, a_V1, a_V2, a_V3

def compute_ss_model(mav, trim_state, trim_input):
    x_euler = euler_state(trim_state)
    return A_lon, B_lon, A_lat, B_lat

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    return x_quat

def f_euler(mav, x_euler, input):
    # return 12x1 dynamics (as if state were Euler state)
    return f_euler_

def df_dx(mav, x_euler, input):
    # take partial of f_euler with respect to x_euler
    return A

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    return (T_eps - T) / eps

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    return (T_eps - T) / eps