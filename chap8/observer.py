"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
#from math import cos, sin, tan
from numpy import cos, sin, tan
from scipy import stats

sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.simulation_parameters as SIM
import parameters.aerosonde_parameters as MAV
import parameters.sensor_parameters as SENSOR
from tools.rotations import Euler2Rotation
from tools.wrap import wrap

from message_types.msg_state import msgState

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msgState()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=0.5)
        self.lpf_gyro_y = alpha_filter(alpha=0.5)
        self.lpf_gyro_z = alpha_filter(alpha=0.5)
        self.lpf_accel_x = alpha_filter(alpha=0.9)
        self.lpf_accel_y = alpha_filter(alpha=0.9)
        self.lpf_accel_z = alpha_filter(alpha=0.9)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9)
        self.lpf_diff = alpha_filter(alpha=0.5)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()

    def update(self, measurements):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x) - SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y) - SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_x) - SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        self.estimated_state.h = self.lpf_static.update(measurements.static_pressure)/(MAV.rho*MAV.gravity)
        self.estimated_state.Va = np.sqrt(2/MAV.rho*self.lpf_diff.update(measurements.diff_pressure))
        if self.estimated_state.Va == 0:
            self.estimated_state.Va = 0.001
        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state

class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        y_prev = self.y
        self.y = self.alpha * y_prev + (1.0 - self.alpha) * u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        noise_param_state = 1e-3
        self.Q = np.diag((1e-9, 1e-9))
        self.Q_gyro = np.eye(3)*SENSOR.gyro_sigma**2
        self.R_accel = np.eye(3)*SENSOR.accel_sigma**2
        self.N = 5  # number of prediction step per sample
        self.xhat = np.vstack((MAV.phi0, MAV.theta0))#np.vstack((MAV.phi0, MAV.theta0)) # initial state: phi, theta
        self.P = np.eye(2)
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state, measurement)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state, measurement):
        # system dynamics for propagation model: xdot = f(x, u)
        p = measurement.gyro_x - state.bx
        q = measurement.gyro_y - state.by
        r = measurement.gyro_z - state.bz
        phi = x[0]
        theta = x[1]
        phidot = p + q*sin(phi)*tan(theta) + r*cos(phi)*tan(theta)
        thetadot = q*cos(phi) - r*sin(phi)
        _f = np.vstack((phidot, thetadot))
        return _f

    def h(self, x, state, measurement):
        # measurement model y
        p = measurement.gyro_x - state.bx
        q = measurement.gyro_y - state.by
        r = measurement.gyro_z - state.bz
        phi = x[0]
        theta = x[1]
        g = MAV.gravity
        Va = state.Va
        accel_x_dot = q*Va*sin(theta) + g*sin((theta))
        accel_y_dot = r*Va*cos(theta) - p*Va*sin(theta) - g*cos(theta)*sin(phi)
        accel_z_dot = -q*Va*cos(theta) - g*cos(theta)*cos(phi)

        _h = np.vstack((accel_x_dot, accel_y_dot, accel_z_dot))
        return _h

    def propagate_model(self, state, measurement):
        # model propagation
        phi = state.phi
        theta = state.theta
        for i in range(0, self.N): 
            # propagate model
            self.xhat += self.Ts * self.f(self.xhat, state, measurement)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state, measurement)
            # compute G matrix for gyro noise, map gyro noise in control space to state space,
            G = np.array([
                [1.0, sin(phi)*tan(theta), cos(phi)*tan(theta)],
                [0.0, cos(phi), -sin(phi)]
                ])
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(len(A)) + A * self.Ts + (A @ A) * (self.Ts**2)/2.0
            G_d = self.Ts * G
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Q * self.Ts**2 + G_d @ self.Q_gyro @ G_d.T

    def measurement_update(self, state, measurement):
        # measurement updates
        h = self.h(self.xhat, state, measurement)
        C = jacobian(self.h, self.xhat, state, measurement)
        y = np.vstack((measurement.accel_x, measurement.accel_y, measurement.accel_z))
        L = self.P @ C.T @ np.linalg.inv(self.R_accel + C @ self.P @ C.T)
        self.xhat = self.xhat + L @ (y - h)
        tmp = np.eye(2) - L @ C
        self.P = tmp @ self.P @ tmp.T + L @ self.R_accel @ L.T

        # S_inv = np.linalg.inv(self.R_accel + C @ self.P @ C.T)
        # if stats.chi2.sf((y-h).T @ S_inv @ (y-h), df=3) > 0.01:
        #     L = self.P @ C.T @ S_inv
        #     self.P = (np.eye(2) - L @ C) @ self.P @ (np.eye(2) - L @ C).T + L @ self.R_accel @ L.T
        #     self.xhat = self.xhat + L @ (y - h)

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        noise_param_state = 1e-10
        self.Q = np.diag((
            1e-5,#pn
            1e-5,#pe
            1e-10,#Vg
            1e-10,#Chi
            10,#wn
            10,#we
            1e-2#psi
        ))
        self.R_gps = np.diag([SENSOR.gps_n_sigma**2, SENSOR.gps_e_sigma**2,  SENSOR.gps_Vg_sigma**2, SENSOR.gps_course_sigma**2,]) 
        self.R_pseudo = np.eye(2)*1e-2
        self.N = 5  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        chi_0 = 0
        self.xhat = np.vstack((MAV.pn0, MAV.pe0, MAV.Va0, chi_0, MAV.w0, MAV.w0, MAV.psi0))#np.vstack((0.0,0.0,25.0,0.0,0.0,0.0,0.0))
        self.P = np.eye(7)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999


    def update(self, state, measurement):
        self.propagate_model(state, measurement)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state, measurement):
        # system dynamics for propagation model: xdot = f(x, u)
        pn = x.item(0)
        pe = x.item(1)
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        p = measurement.gyro_x
        r = measurement.gyro_y
        q = measurement.gyro_z

        pndot = Vg*cos(chi)
        pedot = Vg*sin(chi)
        Va = state.Va        
        theta = state.theta
        phi = state.phi

        chidot = MAV.gravity/Vg*tan(phi)*cos(chi-psi)
        psidot = q*sin(phi)/cos(theta) + r*cos(phi)/cos(theta)
        Vgdot = ((Va*cos(psi) + wn) * (-Va*psidot*sin(psi)) + (Va*sin(psi) + we)*(Va*psidot*cos(psi)))*1/Vg
        _f = np.vstack((pndot, pedot, Vgdot, chidot, 0, 0, psidot))
        return _f

    def h_gps(self, x, state, measurement):
        # measurement model for gps measurements
        _h = np.vstack((x.item(0), x.item(1), x.item(2), x.item(3)))
        return _h

    def h_pseudo(self, x, state, measurement):
        # measurement model for wind triangale pseudo measurement
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = state.Va
        y_wind_tri_n = Va*cos(psi) + wn - Vg*cos(chi)
        y_wind_tri_e = Va*sin(psi) + we - Vg*sin(chi)
        _h = np.vstack((y_wind_tri_n, y_wind_tri_e))
        return _h

    def propagate_model(self, state, measurement):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat += self.Ts * self.f(self.xhat, state, measurement)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state, measurement)
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(len(A)) + A*self.Ts + A @ A * self.Ts**2/2
            # update P with discrete time model
            self.P = A_d @ self.P @ A_d.T + self.Ts**2*self.Q

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, state, measurement)
        C = jacobian(self.h_pseudo, self.xhat, state, measurement)
        y = np.vstack((0, 0))  
        L = self.P @ C.T @ np.linalg.inv(self.R_pseudo + C @ self.P @ C.T)

        self.xhat = self.xhat + L @ (y - h)
        I = np.eye(7)
        tmp = (I - L @ C)
        self.P = tmp @ self.P @ tmp.T + L @ self.R_pseudo @ L.T

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state, measurement)
            C = jacobian(self.h_gps, self.xhat, state, measurement)
            y = np.vstack((measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course))
            L = self.P @ C.T @ np.linalg.inv(self.R_gps + C @ self.P @ C.T)

            y[3,0] = self.wrap(y[3,0], h[3,0])
            self.xhat = self.xhat + L @ (y - h)
            tmp = np.eye(7) - L @ C
            self.P = tmp @ self.P @ tmp.T + L @ self.R_gps @ L.T

            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

    def wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

def jacobian(fun, x, state, measurement):
    # compute jacobian of fun with respect to x
    f = fun(x, state, measurement)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, state, measurement)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J