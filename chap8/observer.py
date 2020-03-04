"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
from math import cos, sin, tan
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
        self.lpf_accel_x = alpha_filter(alpha=0.5)
        self.lpf_accel_y = alpha_filter(alpha=0.5)
        self.lpf_accel_z = alpha_filter(alpha=0.5)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9)
        self.lpf_diff = alpha_filter(alpha=0.5)
        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        self.position_ekf = ekf_position()

    def update(self, measurements):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x) - self.estimated_state.bx #SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y) - self.estimated_state.by#SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_x) - self.estimated_state.bz#SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        self.estimated_state.h = self.lpf_static.update(measurements.static_pressure)/(MAV.rho*MAV.gravity)
        self.estimated_state.Va = np.sqrt(2/MAV.rho*self.lpf_diff.update(measurements.diff_pressure))

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
        self.y = self.alpha * y_prev + (1-self.alpha) * u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        noise_param_state = 1e-10
        self.Q = np.eye(2)*noise_param_state
        self.Q_gyro = np.eye(3)*SENSOR.gyro_sigma**2
        self.R_accel = np.eye(3)*SENSOR.accel_sigma**2
        self.N = 5  # number of prediction step per sample
        self.xhat = np.vstack((0.0, 0.0))#np.vstack((MAV.phi0, MAV.theta0)) # initial state: phi, theta
        self.P = np.eye(2)
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state, measurement)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, measurement, state):
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

    def h(self, x, measurement, state):
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
        for i in range(0, self.N):
            p = state.p
            q = state.q
            r = state.r
            phi = self.xhat[0]
            theta = self.xhat[1]
            # propagate model
            self.xhat += self.Ts * self.f(self.xhat, measurement, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)
            # compute G matrix for gyro noise, map gyro noise in control space to state space,
            G = np.array([
                [1, sin(phi)*tan(theta), cos(phi)*tan(theta)],
                [0, cos(phi), sin(phi)]
                ])
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(len(A)) + A*self.Ts + A @ A.T * self.Ts**2/2
            G_d = self.Ts * G
            # update P with discrete time model
            self.P = self.P + self.Ts * (A_d @ self.P + self.P @ A_d.T + self.Q + G_d @ self.Q_gyro @ G_d.T)

    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([measurement.accel_x, measurement.accel_y, measurement.accel_z])
        for i in range(0, 3):
            if np.abs(y[i]-h[i,0]) < threshold:
                hi = h[i]
                Ci = np.atleast_2d(C[i])
                L = self.P @ Ci.T @ (self.R_accel[i,i] + Ci @ self.P @ Ci.T)**-1
                self.P = (np.eye(2) - L @ Ci) @ self.P @ (np.eye(2) - L @ Ci).T + L @ np.atleast_2d(self.R_accel[i,i]) @ L.T
                self.xhat = self.xhat + L @ np.atleast_2d(y[0] - hi[0])

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        noise_param_state = 1e-10
        self.Q = np.eye(7)*noise_param_state
        self.R = np.diag([SENSOR.gps_n_sigma**2, SENSOR.gps_e_sigma**2,  SENSOR.gps_Vg_sigma**2, SENSOR.gps_course_sigma**2,]) 
        self.R_pseudo = np.eye(2)*1e-5
        self.N = 5  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        chi_0 = 0
        self.xhat = np.vstack((MAV.pn0, MAV.pe0, MAV.Va0, chi_0, MAV.w0, MAV.w0, MAV.psi0))
        self.P = np.eye(7)
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999


    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        pn = x.item(0)
        pe = x.item(1)
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        p = state.p
        r = state.r
        q = state.q

        pndot = Vg*cos(chi)
        pedot = Vg*sin(chi)
        Va = state.Va
        wndot = 0
        wedot = 0
        # Transform state quaternions to theta and phi
        theta = state.theta
        phi = state.phi
        chidot = MAV.gravity/Vg*tan(phi)*cos(chi-psi)
        psidot = q*sin(phi)/cos(theta) + r*cos(phi)/sin(theta)
        Vgdot = ((Va*cos(psi) + wn)*(-Va*psidot*sin(psi)) + (Va*sin(psi) + we)*(Va*psidot*cos(psi)))/Vg
        _f = np.vstack((pndot, pedot, Vgdot, chidot, 0, 0, psidot))
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        _h = np.vstack((x.item(0), x.item(1), x.item(2), x.item(3)))
        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangale pseudo measurement
        pn = x.item(0)
        pe = x.item(1)
        Vg = x.item(2)
        chi = x.item(3)
        wn = x.item(4)
        we = x.item(5)
        psi = x.item(6)
        Va = 0
        y_wind_tri_n = Va*cos(psi) + wn - Vg*cos(chi)
        y_wind_tri_e = Va*sin(psi) + we - Vg*sin(chi)
        _h = np.vstack((y_wind_tri_n, y_wind_tri_e))
        return _h

    def propagate_model(self, measurement, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat += self.Ts * self.f(self.xhat, state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, measurement, state)
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d = np.eye(len(A)) + A*self.Ts + A @ A.T * self.Ts**2/2
            # update P with discrete time model
            self.P = self.P + self.Ts * (A_d @ self.P + self.P @ A_d.T + self.Q)

    def measurement_update(self, measurement, state):
        # always update based on wind triangle pseudo measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, measurement, state)
        y = np.vstack((0, 0))
        P_update = self.P[4:6, 4:6]
        C_update = C[:, 4:6]
        Sinv = np.linalg.inv(self.R_pseudo + C_update @  P_update  @ C_update.T)
        L =  P_update  @ C_update.T @ Sinv
        tmp = np.eye(2) - L @ C_update
        P_update  = tmp @ P_update @ tmp.T + L @ self.R_pseudo @ L.T
        self.xhat[4:6] = self.xhat[4:6] + L @ (y - h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, measurement, state)
            y = np.vstack((measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course))
            P_update = self.P[0:4, 0:4]
            C_update = C[:, 0:4]

            Sinv = np.linalg.inv(self.R + C_update @  P_update  @ C_update.T)
            L =  P_update  @ C_update.T @ Sinv
            tmp = np.eye(4) - L @ C_update
            P_update  = tmp @ P_update @ tmp.T + L @ self.R @ L.T
            self.xhat[0:4] = self.xhat[0:4] + L @ (y - h)
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

def jacobian(fun, x, measurement, state):
    # compute jacobian of fun with respect to x
    f = fun(x, measurement, state)
    m = f.shape[0]
    n = x.shape[0]
    eps = 0.01  # deviation
    J = np.zeros((m, n))
    for i in range(0, n):
        x_eps = np.copy(x)
        x_eps[i][0] += eps
        f_eps = fun(x_eps, state)
        df = (f_eps - f) / eps
        J[:, i] = df[:, 0]
    return J