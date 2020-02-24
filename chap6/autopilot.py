"""
autopilot block for mavsim_python
    - Beard & McLain, PUP, 2012
    - Last Update:
        2/6/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as AP
from tools.transfer_function import TransferFunction
from tools.wrap import wrap
from chap6.pid_control import pid_control, pi_control, pd_control_with_rate
from message_types.msg_state import msg_state


class autopilot:
    def __init__(self, ts_control):
        # instantiate lateral controllers
        self.roll_from_aileron = pd_control_with_rate(
                        kp=AP.roll_kp,
                        kd=AP.roll_kd,
                        limit=np.radians(45))
        self.course_from_roll = pi_control(
                        kp=AP.course_kp,
                        ki=AP.course_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.sideslip_from_rudder = pi_control(
                        kp=AP.sideslip_kp,
                        ki=AP.sideslip_ki,
                        Ts=ts_control,
                        limit=np.radians(45))
        self.yaw_damper = TransferFunction(
                        num=np.array([[AP.yaw_damper_kp, 0]]),
                        den=np.array([[1, 1/AP.yaw_damper_tau_r]]),
                        Ts=ts_control)

        # instantiate longitudinal controllers
        self.pitch_from_elevator = pd_control_with_rate(
                        kp=AP.pitch_kp,
                        kd=AP.pitch_kd,
                        limit=np.radians(45))
        self.altitude_from_pitch = pi_control(
                        kp=AP.altitude_kp,
                        ki=AP.altitude_ki,
                        Ts=ts_control,
                        limit=np.radians(30))
        self.airspeed_from_throttle = pi_control(
                        kp=AP.airspeed_throttle_kp,
                        ki=AP.airspeed_throttle_ki,
                        Ts=ts_control,
                        limit=1.0)
        self.commanded_state = msg_state()

    def update(self, cmd, state):
        # Wrap chi to be within +- pi of the state
        chi_c = wrap(cmd.course_command, state.chi)

        # lateral autopilot
        phi_c_unsaturated = self.course_from_roll.update(chi_c, state.chi)
        phi_c_limit = np.pi/6
        phi_c = self.saturate(phi_c_unsaturated, -phi_c_limit, phi_c_limit) 
        delta_a = self.roll_from_aileron.update(phi_c, state.phi, state.p)
        delta_r = self.yaw_damper.update(state.r)

        # longitudinal autopilot
        h_c = self.saturate(cmd.altitude_command, state.h - AP.altitude_zone, state.h + AP.altitude_zone)
        theta_c = self.altitude_from_pitch.update(h_c, state.h)
        delta_e = self.pitch_from_elevator.update(theta_c, state.theta, state.q)
        delta_t_unsat = self.airspeed_from_throttle.update(cmd.airspeed_command, state.Va)
        delta_t = self.saturate(delta_t_unsat, 0, 1.0)

        # construct output and commanded states
        delta = np.array([[delta_e], [delta_a], [delta_r], [delta_t]])
        self.commanded_state.h = cmd.altitude_command
        self.commanded_state.Va = cmd.airspeed_command
        self.commanded_state.phi = phi_c
        self.commanded_state.theta = theta_c
        self.commanded_state.chi = cmd.course_command
        return delta, self.commanded_state

    def saturate(self, input_, low_limit, up_limit):
        if input_ <= low_limit:
            output = low_limit
        elif input_ >= up_limit:
            output = up_limit
        else:
            output = input_
        return output
