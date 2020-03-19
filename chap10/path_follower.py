import numpy as np
from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msgAutopilot

class path_follower:
    def __init__(self):
        self.chi_inf = np.radians(60)  # approach angle for large distance from straight-line path
        self.k_path = 0.011 # proportional gain for straight-line path following
        self.k_orbit = 2  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msgAutopilot()  # message sent to autopilot

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        chi_q = np.arctan2(path.line_direction.item(1), path.line_direction.item(0))
        chi = state.chi
        chi_q = self._wrap(chi_q, chi)
        Rip = np.array([
            [np.cos(chi_q), np.sin(chi_q), 0],
            [-np.sin(chi_q), np.cos(chi_q), 0],
            [0,0,1]])
        p_i = np.vstack((state.pn, state.pe, -state.h))
        r_i = path.line_origin
        e_p = Rip @ (p_i - r_i)
        ei_p = p_i - r_i
        q = path.line_direction
        ki = np.vstack((0,0,1))
        q_cross_k = np.cross(q.flatten(), ki.flatten())
        unit_vec_norm_to_qk_plane = (q_cross_k/np.linalg.norm(q_cross_k))[:,None]
        si = ei_p - (ei_p @ unit_vec_norm_to_qk_plane.T) * unit_vec_norm_to_qk_plane
        
        self.autopilot_commands.airspeed_command = path.airspeed
        self.autopilot_commands.course_command = chi_q - self.chi_inf*2/np.pi*np.arctan(self.k_path*e_p.item(1))
        sn = si.item(0)
        se = si.item(1)
        qn = q.item(0)
        qe = q.item(1)
        qd = q.item(2)
        self.autopilot_commands.altitude_command = -r_i.item(2) - np.sqrt(sn**2 + se**2)*(qd/(np.sqrt(qn**2 + qe**2)))
        self.autopilot_commands.phi_feedforward = self._calculate_phi_feedforward(path, state, chi)

    def _calculate_phi_feedforward(self, path, state, chi):
        if path.orbit_direction == 'CW':
            lambda_ = 1
        else:
            lambda_ = -1
        wn = state.wn
        we = state.we
        wd = 0
        Va = state.Va
        wn_cos_we_sin = wn*np.cos(chi) + we*np.sin(chi)
        wn_sin_we_cos = wn*np.sin(chi) - we*np.cos(chi)
        sqrt_term = Va**2 - wn_sin_we_cos**2 - wd**2
        R = path.orbit_radius
        g = self.gravity
        return lambda_ * np.arctan2(
            (wn_cos_we_sin + np.sqrt(sqrt_term))**2,
            (g*R*np.sqrt(sqrt_term/(Va**2 - wd**2))))

    def _follow_orbit(self, path, state):
        pn = state.pn
        pe = state.pe
        pd = -state.h
        ce = path.orbit_center.item(1)
        cn = path.orbit_center.item(0)
        cd = path.orbit_center.item(2)
        self.autopilot_commands.airspeed_command = path.airspeed
        chi = state.chi
        psi = np.arctan2(pe-ce, pn-cn)
        hc = -path.orbit_center.item(2)
        psi = self._wrap(psi, chi)
        rho = path.orbit_radius
        d = np.sqrt((pn-cn)**2 + (pe-ce)**2 + (pd-cd)**2)
        if path.orbit_direction == 'CW':
            lambda_ = 1
        else:
            lambda_ = -1
        self.autopilot_commands.course_command = psi + lambda_*(np.pi/2 + np.arctan2(self.k_orbit*(d-rho), rho))
        self.autopilot_commands.altitude_command = hc
        self.autopilot_commands.phi_feedforward = self._calculate_phi_feedforward(path, state, chi)

    def _wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c

