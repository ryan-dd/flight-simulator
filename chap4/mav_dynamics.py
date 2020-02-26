"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
"""
import sys
sys.path.append('..')
import numpy as np
from math import cos, sin

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Quaternion2Euler

class mav_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                               [MAV.pe0],   # (1)
                               [MAV.pd0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        pddots = (np.array([
            [e1**2 + e0**2 - e2**2 - e3**2, 2*(e1*e2 - e3*e0), 2*(e1*e3 + e2*e0)],
            [2*(e1*e2 + e3*e0), e2**2 + e0**2 - e1**2 - e3**3, 2*(e2*e3 - e1*e0)],
            [2*(e1*e3 - e2*e0), 2*(e2*e3 + e1*e0), e3**2 + e0**2 - e1**2 - e2**2]]) @ np.vstack((u,v,w))).flatten()
        # position kinematics
        pn_dot = pddots[0]
        pe_dot = pddots[1]
        pd_dot = pddots[2]

        # position dynamics
        u_dot = (r*v - q*w + 1/MAV.mass * fx)
        v_dot = (p*w - r*u + 1/MAV.mass * fy)
        w_dot = (q*u - p*v + 1/MAV.mass * fz)
        edots = 1/2*np.array([
            [0, -p, -q , -r],
            [p, 0, r, -q],
            [q, -r, 0, p],
            [r, q, -p, 0]]) @ np.vstack((e0, e1, e2, e3)).flatten()
        # rotational kinematics
        e0_dot = edots[0]
        e1_dot = edots[1]
        e2_dot = edots[2]
        e3_dot = edots[3]

        # rotatonal dynamics
        p_dot = MAV.gamma1*p*q - MAV.gamma2*q*r + MAV.gamma3*l+MAV.gamma4*n
        q_dot = MAV.gamma5*p*r - MAV.gamma6*(p**2 - r**2) + 1/MAV.Jy*m
        r_dot = MAV.gamma7*p*q - MAV.gamma1*q*r + MAV.gamma4*l+MAV.gamma8*n 

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]

        # Unpack wind parameters
        wn = wind[0].item(0)
        we = wind[1].item(0)
        wd = wind[2].item(0)
        w_u = wind[3].item(0)
        w_v = wind[4].item(0)
        w_w = wind[5].item(0)
        # Unpack necessary states
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        Vbw = (np.array([
            [e1**2 + e0**2 - e2**2 - e3**2, 2*(e1*e2 - e3*e0), 2*(e1*e3 + e2*e0)],
            [2*(e1*e2 + e3*e0), e2**2 + e0**2 - e1**2 - e3**3, 2*(e2*e3 - e1*e0)],
            [2*(e1*e3 - e2*e0), 2*(e2*e3 + e1*e0), e3**2 + e0**2 - e1**2 - e2**2]]) @ np.vstack((wn,we,wd))).flatten()
        total_wind = Vbw + np.array([w_u, w_v, w_w])
        uw = total_wind[0]
        vw = total_wind[1]
        ww = total_wind[2]
        self._wind[0][0] = uw
        self._wind[1][0] = vw
        self._wind[2][0] = ww
        u = self._state.item(3)
        v = self._state.item(4)
        w = self._state.item(5)
        ur = u-uw
        vr = v-vw
        wr = w-ww
        # compute airspeed
        self._Va = np.sqrt(ur**2 + vr**2 + wr**2)
        # compute angle of attack
        self._alpha = np.arctan2(wr, ur).item(0)
        # compute sideslip angle
        self._beta = np.arcsin(vr/(self._Va)).item(0)

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        # Unpack necessary states
        delta_e = delta[0].item(0)
        delta_t = delta[1].item(0) 
        delta_a = delta[2].item(0) 
        delta_r = delta[3].item(0)
        
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        p = self._state.item(10)
        q = self._state.item(11)
        r = self._state.item(12)

        alpha = self._alpha
        beta = self._beta
        Va = self._Va
        # Calculate fg
        fg = MAV.mass*MAV.gravity*np.vstack((
            (2*(e1*e3 - e2*e0), 
            2*(e2*e3 + e1*e0), 
            e3**2 + e0**2 - e1**2 - e2**2)
        ))
        # Calculate fa
        first = 1/2 * MAV.rho * self._Va**2 * MAV.S_wing
        sigma_a = (1 + np.exp(-MAV.M*(alpha-MAV.alpha0)) + np.exp(MAV.M*(alpha+MAV.alpha0))) / (1 + np.exp(-MAV.M*(alpha-MAV.alpha0))*(1 + np.exp(MAV.M*(alpha+MAV.alpha0))))
        C_L_alpha_f = (1 - sigma_a)*(MAV.C_L_0 + MAV.C_L_alpha*alpha) + sigma_a*(2*np.sign(alpha)*np.sin(alpha)**2*alpha*np.cos(alpha))
        C_D_alpha_f = MAV.C_D_p + (MAV.C_L_0 + MAV.C_L_alpha*alpha)**2/(np.pi*MAV.e*MAV.AR)
        
        C_X_alpha = -C_D_alpha_f*cos(alpha) + C_L_alpha_f*sin(alpha)
        C_X_q_alpha = -MAV.C_D_q*cos(alpha) + MAV.C_L_q*sin(alpha)
        C_X_delta_e = -MAV.C_D_delta_e*cos(alpha) + MAV.C_L_delta_e*sin(alpha)
        C_Z_alpha = -C_D_alpha_f*sin(alpha) - C_L_alpha_f*cos(alpha)
        C_Z_q = -MAV.C_D_q*sin(alpha) - MAV.C_L_q * cos(alpha)
        C_Z_delta_e = -MAV.C_D_delta_e*sin(alpha) - MAV.C_L_delta_e*cos(alpha)

        fx_a = C_X_alpha + C_X_q_alpha*MAV.c/(2*Va)*q + C_X_delta_e*delta_e
        fy_a = MAV.C_Y_0 + MAV.C_Y_beta*beta + MAV.C_Y_p * MAV.b/(2*Va)*p + MAV.C_Y_r*MAV.b/(2*Va)*r + MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r
        fz_a = C_Z_alpha + C_Z_q*MAV.c/(2*Va)*q + C_Z_delta_e*delta_e
        fa = first*np.vstack((fx_a, fy_a, fz_a))

        # Calculate Fp and Mp
        # compute t h r u s t and torque due to p r o p ell e r ( See addendum by McLain)
        # map d e l t a t t h r o t t l e command(0 t o 1) i n t o motor i n p u t v o l t a g e
        V_in = MAV.V_max*delta_t
        # Quadratic formula to solve for motor speed
        a_part = MAV.C_Q0 * MAV.rho * np.power(MAV.D_prop, 5)/((2.*np.pi)**2)
        b_part = (MAV.C_Q1 * MAV.rho * np.power(MAV.D_prop, 4)
        / (2.*np.pi))*self._Va + MAV.KQ**2/MAV.R_motor
        c_part = MAV.C_Q2 * MAV.rho * np.power(MAV.D_prop, 3)*self._Va**2 - (MAV.KQ/MAV.R_motor)*V_in + MAV.KQ * MAV.i0
        # Consider only positive root
        Omega_op = (-b_part + np.sqrt(b_part**2 - 4*a_part*c_part))/(2.*a_part)
        # compute advance rat io
        J_op = 2*np.pi*self._Va/(Omega_op*MAV.D_prop)
        # compute nond imens ional ized c o e f f i c i e n t s of thrus t and torque
        C_T = MAV.C_T2*J_op**2 + MAV.C_T1*J_op + MAV.C_T0
        C_Q = MAV.C_Q2*J_op**2 + MAV.C_Q1*J_op + MAV.C_Q0
        # add thrus t and torque due to pr o peller
        n_input = Omega_op/(2*np.pi)
    
        f = fa + fg
        fx = f[0][0]
        fy = f[1][0]
        fz = f[2][0]

        Mx = MAV.b*(MAV.C_ell_0 + MAV.C_ell_beta*beta + MAV.C_ell_p * MAV.b/(2*Va)*p + MAV.C_ell_r*MAV.b/(2*Va)*r + MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r)
        My = MAV.c*(MAV.C_m_0 + MAV.C_m_alpha*alpha + MAV.C_m_q*MAV.c/(2*Va)*q + MAV.C_m_delta_e*delta_e)
        Mz = MAV.b*(MAV.C_n_0 + MAV.C_n_beta*beta + MAV.C_n_p*MAV.b/(2*Va)*p + MAV.C_n_r*MAV.b/(2*Va)*r + MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r)
        
        Mx = Mx*first
        My = My*first
        Mz = Mz*first
        fx += MAV.rho*n_input**2*np.power(MAV.D_prop, 4)*C_T
        Mx += -MAV.rho*n_input**2*np.power(MAV.D_prop, 5)*C_Q

        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz
        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi.item(0)
        self.msg_true_state.theta = theta.item(0)
        self.msg_true_state.psi = psi.item(0)
        u = self._state[3]
        v = self._state[4]
        w = self._state[5]
        e0 = self._state[6]
        e1 = self._state[7]
        e2 = self._state[8]
        e3 = self._state[9]

        Vg = np.sqrt(u**2 + v**2 + w**2).item(0)
    
        self.msg_true_state.Vg = Vg
        gamma_wt = np.arcsin(-w/Vg)
        self.msg_true_state.gamma = gamma_wt.item(0)
        self.msg_true_state.chi = np.arctan2(v, u).item(0)
        self.msg_true_state.p = self._state[10].item(0)
        self.msg_true_state.q = self._state[11].item(0)
        self.msg_true_state.r = self._state[12].item(0)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)
