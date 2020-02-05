"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
from tools.transfer_function import TransferFunction
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts

def compute_tf_model(mav, trim_state, trim_input):
    # trim values
    aphi1 = -1/2*MAV.rho*mav._Va**2 * MAV.S_wing*MAV.b*MAV.C_p_p * MAV.b/(2*mav._Va)
    aphi2 = 1/2*MAV.rho*mav._Va**2 * MAV.S_wing*MAV.b*MAV.C_p_delta_a
    T_phi_delta_a = TransferFunction(np.array([[aphi2]]), np.array([[1, aphi1, 0]]), Ts)

    T_chi_phi = TransferFunction(np.array([[MAV.gravity/mav._Va]]), np.array([[1]]), Ts)

    abeta1 = -MAV.rho*mav._Va*MAV.S_wing * MAV.C_Y_beta / (2*MAV.mass)
    abeta2 = MAV.rho*mav._Va*MAV.S_wing * MAV.C_Y_delta_r / (2*MAV.mass)
    T_beta_delta_r = TransferFunction(np.array([[abeta2]]), np.array([[1, abeta1]]), Ts)

    atheta1 = -MAV.rho*mav._Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_q * MAV.c/(2*mav._Va)
    atheta2 = -MAV.rho*mav._Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_alpha
    atheta3 = MAV.rho*mav._Va**2 * MAV.c*MAV.S_wing/(2*MAV.Jy) * MAV.C_m_delta_e
    T_theta_delta_e =TransferFunction(np.array([[atheta3]]), np.array([[1, atheta1, atheta2]]), Ts)

    T_h_theta = TransferFunction(np.array([[mav._Va]]), np.array([[1]]), Ts)

    _, theta, _ = Quaternion2Euler(mav._state[5:9])
    T_h_Va = TransferFunction(np.array([[theta]]), np.array([[1]]), Ts)


    Va_star = trim_state._
    alpha_star = MAV.alpha0
    delta_e_star = trim_input[0].item(0)
    delta_t_star = trim_input[1].item(0) 

    aV1 = MAV.rho*Va_star*MAV.S_wing/MAV.mass * (MAV.C_D_0 + MAV.C_D_alpha*alpha_star + MAV.C_D_delta_e * delta_e_star) + MAV.rho * MAV.S_prop/MAV.mass * MAV.C_prop*Va_star
    aV2 = MAV.rho * MAV.S_prop/MAV.mass * MAV.C_prop*MAV.k_motor**2*delta_t_star
    aV3 = MAV.gravity
    T_Va_delta_t = TransferFunction(np.array([[aV2]]), np.array([[1, aV1]]), Ts)
    T_Va_theta = TransferFunction(np.array([[-aV3]]), np.array([[1, aV1]]), Ts)

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r

def compute_ss_model(mav, trim_state, trim_input):
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
    # compute f at euler_state
    return f_euler_

def df_dx(mav, x_euler, input):
    # take partial of f_euler with respect to x_euler
    return A

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    return dThrust

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    return dThrust