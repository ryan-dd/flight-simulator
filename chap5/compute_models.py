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
    
    T_phi_delta_a = TransferFunction(np.array([[aphi2]]), np.array([[1, aphi1, 0]]), Ts)

    T_chi_phi = TransferFunction(np.array([[MAV.gravity/mav._Va]]), np.array([[1, 0]]), Ts)

    
    T_beta_delta_r = TransferFunction(np.array([[abeta2]]), np.array([[1, abeta1]]), Ts)

    
    T_theta_delta_e =TransferFunction(np.array([[atheta3]]), np.array([[1, atheta1, atheta2]]), Ts)

    T_h_theta = TransferFunction(np.array([[mav._Va]]), np.array([[1, 0]]), Ts)

    _, theta, _ = Quaternion2Euler(mav._state[5:9])
    T_h_Va = TransferFunction(np.array([[theta]]), np.array([[1, 0]]), Ts)


    Va_star = trim_state[3]
    alpha_star = MAV.alpha0
    delta_e_star = trim_input[0].item(0)
    delta_t_star = trim_input[1].item(0) 

    
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
    alpha_star = 0
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
    sec = lambda a : 1/np.cos(a)
    A_lat = np.array([
        [Yv, Yp, Yr, MAV.gravity*np.cos(thetastar)*np.cos(phistar), 0],
        [Lv, Lp, Lr, 0, 0],
        [Nv, Np, Nr, 0, 0],
        [0, 1, np.cos(phistar)*np.tan(thetastar), qstar*np.cos(phistar)*np.tan(thetastar) - rstar*np.sin(phistar)*np.tan(thetastar), 0],
        [0, 0, np.cos(phistar)*sec(thetastar), pstar*np.cos(phistar)*sec(thetastar) - rstar*np.sin(phistar)*sec(thetastar), 0]
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
    
    
    C_X_q = 0
    C_X_alpha = -C_D_alpha_f*cos(alpha) + C_L_alpha_f*sin(alpha)
    C_X_q_alpha = -MAV.C_D_q*cos(alpha) + MAV.C_L_q*sin(alpha)
    C_X_delta_e = -MAV.C_D_delta_e*cos(alpha) + MAV.C_L_delta_e*sin(alpha)
    C_Z_0 = 0
    C_Z_alpha = -C_D_alpha_f*sin(alpha) - C_L_alpha_f*cos(alpha)
    C_Z_q = -MAV.C_D_q*sin(alpha) - MAV.C_L_q * cos(alpha)
    C_Z_delta_e = -MAV.C_D_delta_e*sin(alpha) - MAV.C_L_delta_e*cos(alpha)
    Xu = ustar*MAV.rho*MAV.S_wing/(MAV.mass)*(C_X_alpha*alpha_star + MAV.C_X_delta_e*delta_e_star) - MAV.rho*MAV.S_wing*wstar*C_X_alpha/(2*MAV.mass) + MAV.rho*MAV.S_wing*MAV.c*C_X_q*ustar*qstar/(4*MAV.mass*Vastar) - MAV.rho*MAV.S_prop*MAV.C_prop*ustar/(MAV.mass)
    Xw = -qstar + wstar*MAV.rho*MAV.S_wing/(MAV.mass)*(C_X_q_alpha*alpha_star+C_X_delta_e*delta_e_star) + MAV.rho*MAV.S_wing*MAV.c*C_X_q*wstar*qstar/(4*MAV.mass*Vastar) + MAV.rho*MAV.S_wing*ustar*C_X_alpha/(2*MAV.mass) - MAV.rho*MAV.S_prop*MAV.C_prop*wstar/(MAV.mass)
    Xq = -wstar + MAV.rho*Vastar * MAV.S_wing*C_X_q*MAV.c/(4*MAV.mass)
    X_delta_e = MAV.rho*MAV.Vastar**2*MAV.S_wing*C_X_delta_e/(2*MAV.mass)
    X_delta_t = MAV.rho*MAV.S_prop*MAV.C_prop*MAV.k_motor**2*delta_t_star/(MAV.mass)
    Zu = qstar + ustar*MAV.rho*MAV.S_wing/(MAV.mass)*(C_Z_0+C_Z_alpha*alpha_star + C_Z_delta_e*delta_e_star) - MAV.rho*MAV.S_wing*C_Z_alpha*wstar/(2*MAV.mass) + ustar*MAV.rho*MAV.S_wing*C_Z_q*MAV.c*qstar/(4*MAV.mass*Vastar)
    Zw = wstar*MAV.rho*MAV.S_wing/(MAV.mass)*(C_Z_0+C_Z_alpha*alpha_star + C_Z_delta_e*delta_e_star) + MAV.rho*MAV.S_wing*C_Z_alpha*ustar/(2*MAV.mass) + MAV.rho*wstar*MAV.S_wing*MAV.c*C_Z_q*qstar/(4*MAV.mass*Vastar)
    Zq = ustar + MAV.rho*Vastar*MAV.S_wing*C_Z_q*MAV.c/(2*MAV.mass)
    Z_delta_e = MAV.rho*MAV.Vastar**2*MAV.S_wing*C_Z_delta_e/(2*MAV.mass)
    M_u = ustar*MAV.rho*MAV.S_wing*MAV.c/(MAV.Jy)*(MAV.C_m_0 + MAV.C_m_alpha*alpha_star + MAV.C_m_delta_e*delta_e_star) - MAV.rho*MAV.S_wing*MAV.c*MAV.C_m_alpha*wstar/(2*MAV.Jy) + MAV.rho*MAV.S_wing*MAV.c**2*MAV.C_m_q*qstar*ustar/(4*MAV.Jy*Vastar)
    M_w = wstar*MAV.rho*MAV.S_wing*MAV.c/(MAV.Jy)*(MAV.C_m_0 + MAV.C_m_alpha*alpha_star + MAV.C_m_delta_e*delta_e_star) + MAV.rho*MAV.S_wing*MAV.c*MAV.C_m_alpha*ustar/(2*MAV.Jy) + MAV.rho*MAV.S_wing*MAV.c**2*MAV.C_m_q*qstar*wstar/(4*MAV.Jy*Vastar)
    Mq = MAV.rho*Vastar*MAV.S_wing*MAV.c**2*MAV.C_m_q/(4*MAV.Jy)
    M_delta_e = MAV.rho*Vastar**2*MAV.S_wing*MAV.c*MAV.C_m_delta_e/(2*MAV.Jy)

    A_lon = np.array([
        [Xu, Xw, Xq, -MAV.gravity*np.cos(thetastar), 0],
        [Zu, Zw, Zq, -MAV.gravity*np.sin(thetastar),0]
        [0, 0, 1, 0, 0],
        [np.sin(thetastar), -np.cos(thetastar), 0, ustar*cos(thetastar)+wstar*np.sin(thetastar), 0]
    ])
    B_lon = np.array([
        [X_delta_e, X_delta_t],
        [Z_delta_e, 0],
        [M_delta_e, 0],
        [0, 0],
        [0, 0]
    ])
    return A_lon, B_lon, A_lat, B_lat

# def euler_state(x_quat):
#     # convert state x with attitude represented by quaternion
#     # to x_euler with attitude represented by Euler angles
#      return x_euler

# def quaternion_state(x_euler):
#     # convert state x_euler with attitude represented by Euler angles
#     # to x_quat with attitude represented by quaternions
#     return x_quat

# def f_euler(mav, x_euler, input):
#     # return 12x1 dynamics (as if state were Euler state)
#     # compute f at euler_state
#     return f_euler_

# def df_dx(mav, x_euler, input):
#     # take partial of f_euler with respect to x_euler
#     return A

# def df_du(mav, x_euler, delta):
#     # take partial of f_euler with respect to delta
#     return B

# def dT_dVa(mav, Va, delta_t):
#     # returns the derivative of motor thrust with respect to Va
#     return dThrust

# def dT_ddelta_t(mav, Va, delta_t):
#     # returns the derivative of motor thrust with respect to delta_t
#     return dThrust