"""
mavDynamics 
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state
    
mavsim_python
    - Beard & McLain, PUP, 2012
    - Update history:  
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np

# load message types
from message_types.msg_state import msgState
from message_types.msg_sensors import msgSensors

import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.rotations import Quaternion2Rotation, Quaternion2Euler, Euler2Rotation

class mavDynamics:
    def __init__(self, Ts):
        # OLD STUFF here
        # initialize the sensors message
        self._sensors = msgSensors()
        # random walk parameters for GPS
        self._gps_eta_n = 0.
        self._gps_eta_e = 0.
        self._gps_eta_h = 0.
        # timer so that gps only updates every ts_gps seconds
        self._t_gps = 999.  # large value ensures gps updates at initial time.
        # update velocity data and forces and moments
        self._update_velocity_data()
        self._forces_moments(delta=np.array([[0.0], [0.0], [0.0], [0.5]]))


    ###################################
    # public functions
    def update(self, delta, wind):
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
        self._update_true_state()

    def sensors(self):
        "Return value of sensors on MAV: gyros, accels, static_pressure, dynamic_pressure, GPS"
        self.sensors.gyro_x = p + SENSOR.gyro_x_bias + np.random.normal(0, SENSOR.gyro_sigma)
        self.sensors.gyro_y = q + SENSOR.gyro_y_bias + np.random.normal(0, SENSOR.gyro_sigma)
        self.sensors.gyro_z = r + SENSOR.gyro_z_bias + np.random.normal(0, SENSOR.gyro_sigma)
        self.sensors.accel_x = udot + q*w - r*v + g*sin(theta) + np.random.normal(0, SENSOR.accel_sigma)
        self.sensors.accel_y = vdot + r*u - p*2 - g*cos(theta)*sin(phi) + np.random.normal(0, SENSOR.accel_sigma)
        self.sensors.accel_z = wdot + p*v - q*u - g*cos(theta)*cos(phi) + np.random.normal(0, SENSOR.accel_sigma)
        beta_abs_pres = 0
        beta_diff_pres = 0
        self.sensors.static_pressure = MAV.rho*MAV.gravity*h_AGL + beta_abs_pres + np.random.normal(0, SENSOR.static_pres_sigma)
        self.sensors.diff_pressure = MAV.rho*self._Va**2/2 + beta_diff_pres + np.random.normal(0, SENSOR.diff_pres_sigma)
        if self._t_gps >= SENSOR.ts_gps:
            self._gps_eta_n = np.exp(-SENSOR.gps_beta*SENSOR.ts_gps)* self._gps_eta_n + np.random.normal(0, SENSOR.gps_n_sigma)
            self._gps_eta_e = np.exp(-SENSOR.gps_beta*SENSOR.ts_gps)* self._gps_eta_e + np.random.normal(0, SENSOR.gps_e_sigma)
            self._gps_eta_h = np.exp(-SENSOR.gps_beta*SENSOR.ts_gps)* self._gps_eta_h + np.random.normal(0, SENSOR.gps_h_sigma)
            self.sensors.gps_n = self._state.item(0) + self._gps_eta_n
            self.sensors.gps_e = self._state.item(1) + self._gps_eta_e
            self.sensors.gps_h = -self._state.item(2) + self._gps_eta_h
            self.sensors.gps_Vg = np.linalg.norm(self._state[3:6]) + np.random.normal(0, SENSOR.gps_Vg_sigma)
            self.sensors.gps_course = np.arctan2(self._state[1], self._state[2]) + np.random.normal(0, SENSOR.gps_course_sigma)
            self._t_gps = 0
        else:
            self._t_gps += self._ts_simulation
        return self._sensors

    def external_set_state(self, new_state):
        self._state = new_state

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):

    def _forces_moments(self, delta):
        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def _motor_thrust_torque(self, Va, delta_t):
         return T_p, Q_p


    def _update_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        pdot = Quaternion2Rotation(self._state[6:10]) @ self._state[3:6]
        self.true_state.pn = self._state.item(0)
        self.true_state.pe = self._state.item(1)
        self.true_state.h = -self._state.item(2)
        self.true_state.Va = self._Va
        self.true_state.alpha = self._alpha
        self.true_state.beta = self._beta
        self.true_state.phi = phi
        self.true_state.theta = theta
        self.true_state.psi = psi
        self.true_state.Vg = np.linalg.norm(pdot)
        self.true_state.gamma = np.arcsin(pdot.item(2) / self.true_state.Vg)
        self.true_state.chi = np.arctan2(pdot.item(1), pdot.item(0))
        self.true_state.p = self._state.item(10)
        self.true_state.q = self._state.item(11)
        self.true_state.r = self._state.item(12)
        self.true_state.wn = self._wind.item(0)
        self.true_state.we = self._wind.item(1)
        self.true_state.bx = SENSOR.gyro_x_bias
        self.true_state.by = SENSOR.gyro_y_bias
        self.true_state.bz = SENSOR.gyro_z_bias
