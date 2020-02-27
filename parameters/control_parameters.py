import sys
sys.path.append('..')
import numpy as np
import chap5.model_coef as TF
import parameters.aerosonde_parameters as MAV

gravity = MAV.gravity  # gravity constant
rho = MAV.rho  # density of air
# sigma =1   # low pass filter gain for derivative
Va0 = TF.Va_trim
Va = Va0
Vg = Va
delta_a_max = np.radians(45)
delta_e_max = np.radians(45)

#----------roll loop-------------
# get transfer function data for delta_a to phi
zeta_roll = 0.707
roll_e_max = np.radians(45)

roll_kp = delta_a_max / roll_e_max
wn_roll = np.sqrt(np.abs(TF.a_phi2)*roll_kp)

roll_kd = (2*zeta_roll*wn_roll - TF.a_phi1)/(TF.a_phi2)

#----------course loop-------------
Wx = 10
wn_course = 1/Wx*wn_roll
zeta_course = 0.707

course_kp = 2*zeta_course*wn_course*Vg/MAV.gravity
course_ki = wn_course**2*Vg/MAV.gravity

#----------sideslip loop-------------
# How to tune emaxbeta?
# e_max_beta = 40
# zeta_beta = 0.707
# delta_r_max = 30

#----------yaw damper-------------
yaw_damper_tau_r = 0.05
yaw_damper_kp = -0.005

#----------pitch loop-------------
e_max_pitch = np.radians(30)
zeta_pitch = 3.0

pitch_kp = delta_e_max/e_max_pitch*np.sign(TF.a_theta3)
wn_pitch = np.sqrt(TF.a_theta2 + pitch_kp*TF.a_theta3)
pitch_kd = (2*zeta_pitch*wn_pitch - TF.a_theta1)/TF.a_theta3
K_theta_DC = (pitch_kp*TF.a_theta3)/(TF.a_theta2 + pitch_kp*TF.a_theta3)


#----------altitude loop-------------
Wh = 10
wn_altitude = 1/Wh*wn_pitch
zeta_altitude = 2.707
altitude_kp = 2*zeta_altitude*wn_altitude/(K_theta_DC*Va)
altitude_ki = wn_altitude**2/(K_theta_DC*Va)
altitude_zone = 2 # moving saturation limit around current altitude

#---------airspeed hold using throttle---------------
wn_throttle = 5
zeta_throttle = 1.707
airspeed_throttle_kp = 1.1#wn_throttle**2/TF.a_V2
airspeed_throttle_ki = 0.35#(2*zeta_throttle*wn_throttle - TF.a_V1)/TF.a_V2
