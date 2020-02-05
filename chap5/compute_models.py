"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from math import sin, cos, tan
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


    Va_star = trim_state[3]
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
    ustar = trim_state.item(3)
    vstar = trim_state.item(4)
    wstar = trim_state.item(5)
    e = trim_state[5:9]
    phistar, thetastar, psistar = Quaternion2Euler(e)
    pstar = trim_state.item(10)
    qstar = trim_state.item(11)
    rstar = trim_state.item(12)
    delta_e_star = trim_input[0].item(0)
    delta_t_star = trim_input[1].item(0) 
    delta_a_star = trim_input[2].item(0) 
    delta_r_star = trim_input[3].item(0)
    betastar = 0
    Vastar = ustar
    Yv = MAV.rho*MAV.S_wing*MAV.b*vstar/(4*MAV.mass*Vastar)*(MAV.C_Y_p*pstar + MAV.C_Y_r*rstar) + MAV.rho* MAV.S_wing*vstar/(MAV.mass) * (
        MAV.C_Y_0 + MAV.C_Y_beta*betastar + MAV.C_Y_delta_a*delta_a_star + MAV.C_Y_delta_r*rstar) + MAV.rho*MAV.S_wing*MAV.C_Y_beta/(2*MAV.mass)*np.sqrt(ustar**2 + wstar**2)
    Yp = wstar + MAV.rho*Vastar*MAV.S_wing*MAV.b/(4*MAV.mass) * MAV.C_Y_p
    Yr = -ustar + MAV.rho*Vastar*MAV.S_wing*MAV.b/(4*MAV.mass) * MAV.C_Y_r
    Y_delta_a = MAV.rho*Vastar**2 * MAV.S_wing/(2*MAV.mass) * MAV.C_Y_delta_a
    Y_delta_r = MAV.rho*Vastar**2 * MAV.S_wing/(2*MAV.mass) * MAV.C_Y_delta_r

    Lv = MAV.rho*MAV.S_wing*MAV.b**2 * vstar/(4*Vastar)*(MAV.C_p_p*pstar + MAV.C_p_r*rstar) + MAV.rho*MAV.S_wing*MAV.b*vstar * (
        MAV.C_p_0 + MAV.C_p_beta*betastar + MAV.C_p_delta_a*delta_a_star + MAV.C_p_delta_r*rstar) + MAV.rho*MAV.S_wing*MAV.b*MAV.C_p_beta/2*np.sqrt(ustar**2+wstar**2)
    Lp = MAV.gamma1*qstar + MAV.rho*Vastar*MAV.S_wing*MAV.b**2/4*MAV.C_p_p
    Lr = -MAV.gamma2*qstar + MAV.rho*Vastar*MAV.S_wing*MAV.b**2/4*MAV.C_p_r
    L_delta_a = MAV.rho*Vastar**2*MAV.S_wing*MAV.b/2*MAV.C_p_delta_a
    L_delta_r = MAV.rho*Vastar**2*MAV.S_wing*MAV.b/2*MAV.C_p_delta_r

    Nv = MAV.rho*MAV.S_wing*MAV.b**2 * vstar/(4*Vastar)*(MAV.C_r_p*pstar + MAV.C_r_r*rstar) + MAV.rho*MAV.S_wing*MAV.b*vstar * (
        MAV.C_r_0 + MAV.C_r_beta*betastar + MAV.C_r_delta_a*delta_a_star + MAV.C_r_delta_r*rstar) + MAV.rho*MAV.S_wing*MAV.b*MAV.C_r_beta/2*np.sqrt(ustar**2+wstar**2)
    Np = MAV.gamma7*qstar + MAV.rho*Vastar*MAV.S_wing*MAV.b**2/4*MAV.C_r_p
    Nr = -MAV.gamma1*qstar + MAV.rho*Vastar*MAV.S_wing*MAV.b**2/4*MAV.C_r_r
    N_delta_a = MAV.rho*Vastar**2*MAV.S_wing*MAV.b/2*MAV.C_r_delta_a
    N_delta_r = MAV.rho*Vastar**2*MAV.S_wing*MAV.b/2*MAV.C_r_delta_r

    A_lat = np.array([
        [Yv, Yp, Yr, MAV.gravity*np.cos(thetastar)*np.cos(phistar), 0],
        [Lv, Lp, Lr, 0, 0],
        [Nv, Np, Nr, 0, 0],
        [0, 1, np.cos(phistar)*np.tan(thetastar), qstar*np.cos(phistar)*np.tan(thetastar) - rstar*np.sin(phistar)*np.tan(thetastar), 0],
        [0, 0, np.cos(phistar)*np.sec(thetastar), pstar*np.cos(phistar)*np.sec(thetastar) - rstar*np.sin(phistar)*np.sec(thetastar), 0]
        ])
    B_lat = np.array([
        [Y_delta_a, Y_delta_r],
        [L_delta_a, L_delta_r],
        [N_delta_a, N_delta_r],
        [0,0],
        [0,0]])
    # Longitudinal dynamics
    sigma_a = (1 + np.exp(-MAV.M*(alpha-MAV.alpha0)) + np.exp(MAV.M*(alpha+MAV.alpha0))) / (1 + np.exp(-MAV.M*(alpha-MAV.alpha0))*(1 + np.exp(MAV.M*(alpha+MAV.alpha0))))
    C_L_alpha_f = (1 - sigma_a)*(MAV.C_L_0 + MAV.C_L_alpha*alpha) + sigma_a*(2*np.sign(alpha)*np.sin(alpha)**2*alpha*np.cos(alpha))
    C_D_alpha_f = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha*alpha)**2/(np.pi*MAV.e*MAV.AR)
    
    
    C_X_alpha = -C_D_alpha_f*cos(alpha) + C_L_alpha_f*sin(alpha)
    C_X_q_alpha = -MAV.C_D_q*cos(alpha) + MAV.C_L_q*sin(alpha)
    C_X_delta_e = -MAV.C_D_delta_e*cos(alpha) + MAV.C_L_delta_e*sin(alpha)
    C_Z_alpha = -C_D_alpha_f*sin(alpha) - C_L_alpha_f*cos(alpha)
    C_Z_q = -MAV.C_D_q*sin(alpha) - MAV.C_L_q * cos(alpha)
    C_Z_delta_e = -MAV.C_D_delta_e*sin(alpha) - MAV.C_L_delta_e*cos(alpha)
    Xu = ustar*MAV.rho*MAV.S_wing/(MAV.mass)*(MAV.C_)
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