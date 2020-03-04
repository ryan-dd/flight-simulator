"""
mavsim_python
    - Chapter 8 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        2/21/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mavViewer
from chap3.data_viewer import dataViewer
from chap4.wind_simulation import windSimulation
from chap6.autopilot import autopilot
from chap7.mav_dynamics import mavDynamics
from chap8.observer import observer
from tools.signals import signals

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = mavViewer()  # initialize the mav viewer
data_view = dataViewer()  # initialize view of data plots
if VIDEO == True:
    from chap2.video_writer import video_writer
    video = video_writer(video_name="chap8_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = windSimulation(SIM.ts_simulation)
mav = mavDynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)
obsv = observer(SIM.ts_simulation)

# autopilot commands
from message_types.msg_autopilot import msgAutopilot
commands = msgAutopilot()
Va_command = signals(dc_offset=25.0, amplitude=3.0, start_time=2.0, frequency = 0.01)
h_command = signals(dc_offset=100.0, amplitude=10.0, start_time=0.0, frequency = 0.02)
chi_command = signals(dc_offset=np.radians(180), amplitude=np.radians(45), start_time=5.0, frequency = 0.015)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:

    #-------autopilot commands-------------
    commands.airspeed_command = 25#Va_command.square(sim_time)
    commands.course_command = 0#chi_command.square(sim_time)
    commands.altitude_command = 100#h_command.square(sim_time)

    #-------controller-------------
    measurements = mav.sensors()  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements
    
    delta, commanded_state = ctrl.update(commands, mav.state)

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    mav_view.update(mav.true_state)  # plot body of MAV
    data_view.update(mav.true_state, # true states
                     estimated_state, # estimated states
                     commanded_state, # commanded states
                     SIM.ts_simulation)
    if VIDEO == True: video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True: video.close()




