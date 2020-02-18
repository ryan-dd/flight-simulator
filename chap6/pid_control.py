"""
pid_control
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')

class pid_control:
    def __init__(self, kp=0.0, ki=0.0, kd=0.0, Ts=0.01, sigma=0.05, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.Ts = Ts
        self.limit = limit
        self.integrator = 0.0
        self.error_delay_1 = 0.0
        self.error_dot_delay_1 = 0.0
        # gains for differentiator
        self.a1 = (2.0 * sigma - Ts) / (2.0 * sigma + Ts)
        self.a2 = 2.0 / (2.0 * sigma + Ts)

    def update(self, y_ref, y, reset_flag=False):
        if reset_flag:
            self.integrator = 0
        error = y_ref - y
        p_term = error*self.kp
        self.integrator += (error - self.error_delay_1)*self.Ts/2
        i_term = self.integrator*self.ki
        error_dot = self.a1*self.error_dot_delay_1 + self.a2*(error - self.error_delay_1)
        d_term = error_dot*self.kd
        self.error_dot_delay_1 = error_dot
        self.error_delay_1 = error
        u = p_term + i_term + d_term
        u_sat = self._saturate(u)
        return u_sat

    def update_with_rate(self, y_ref, y, ydot, reset_flag=False):
        if reset_flag:
            self.integrator = 0
        error = y_ref - y
        p_term = error*self.kp
        self.integrator += (error + self.error_delay_1)*self.Ts/2
        i_term = self.integrator*self.ki
        d_term = ydot*self.kd
        u = p_term + i_term + d_term
        u_sat = self._saturate(u)
        if self.ki > 0.00001:
            if not np.isclose(u, u_sat):
                self.integrator += 1/self.ki*(self.limit-u)
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat

class pi_control:
    def __init__(self, kp=0.0, ki=0.0, Ts=0.01, limit=1.0):
        self.kp = kp
        self.ki = ki
        self.Ts = Ts
        self.limit = limit
        self.integrator = 0.0
        self.error_delay_1 = 0.0

    def update(self, y_ref, y):
        error = y_ref - y
        p_term = error*self.kp
        self.integrator += (error + self.error_delay_1)*self.Ts/2
        i_term = self.integrator*self.ki
        u = p_term + i_term
        u_sat = self._saturate(u)
        if self.ki > 0.00001:
            if not np.isclose(u, u_sat):
                self.integrator += 1/self.ki*(self.limit-u)
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat

class pd_control_with_rate:
    # PD control with rate information
    # u = kp*(yref-y) - kd*ydot
    def __init__(self, kp=0.0, kd=0.0, limit=1.0):
        self.kp = kp
        self.kd = kd
        self.limit = limit

    def update(self, y_ref, y, ydot):
        error = y_ref - y
        p_term = error*self.kp
        d_term = ydot*self.kd
        u = p_term + d_term
        u_sat = self._saturate(u)
        return u_sat

    def _saturate(self, u):
        # saturate u at +- self.limit
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat