"""
mavsimPy
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mav_viewer
from chap3.data_viewer import data_viewer
from chap3.mav_dynamics import mav_dynamics


# initialize the visualization
VIDEO = True  # True==write video, False==don't write video
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots
if VIDEO == True:
    from chap2.video_writer import VideoWriter
    video = VideoWriter(video_name="chap3_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
mav = mav_dynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:
    #-------vary forces and moments to check dynamics-------------
    if sim_time < 5:
        fx = 10
        fy = 0 # 10
        fz = 0 # 10
        Mx = 0 # 0.1
        My = 0 # 0.1
        Mz = 0 # 0.1
    elif sim_time < 10:
        fx = -20
        fy = 40 # 10
        fz = 0 # 10
        Mx = 0 # 0.1
        My = 0 # 0.1
        Mz = 0 # 0.1
    elif sim_time < 15:
        fx = 0
        fy = -40 # 10
        fz = 30 # 10
        Mx = 0 # 0.1
        My = 0 # 0.1
        Mz = 0 # 0.1
    elif sim_time < 20:
        fx = 0
        fy = 0 # 10
        fz = 0 # 10
        Mx = 0.1 # 0.1
        My = 0 # 0.1
        Mz = 0 # 0.1
    elif sim_time < 25:
        fx = 0
        fy = 0 # 10
        fz = 0 # 10
        Mx = 0 # 0.1
        My = 0.3 # 0.1
        Mz = 0 # 0.1
    elif sim_time < 30:
        fx = 0
        fy = 0 # 10
        fz = 0 # 10
        Mx = 0 # 0.1
        My = 0 # 0.1
        Mz = 0.4# 0.1
    elif sim_time < 35:
        fx = 10
        fy = 0 # 10
        fz = 0 # 10
        Mx = 0.4 # 0.1
        My = 0 # 0.1
        Mz = 0 # 0.1
    elif sim_time < 40:
        fx = 0
        fy = 10 # 10
        fz = 0 # 10
        Mx = 0 # 0.1
        My = 0.3 # 0.1
        Mz = 0 # 0.1
    elif sim_time < 45:
        fx = 0
        fy = 0 # 10
        fz = 0 # 10
        Mx = 0 # 0.1
        My = 0 # 0.1
        Mz = 0.1 # 0.1
    elif sim_time < 50:
        fx = -30
        fy = -30 # 10
        fz = -30 # 10
        Mx = 0.1 # 0.1
        My = 0.1 # 0.1
        Mz = 0.1 # 0.1

    forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    #-------physical system-------------
    mav.update_state(forces_moments)  # propagate the MAV dynamics

    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)
    if VIDEO == True: video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True: video.close()




