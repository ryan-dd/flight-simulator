"""
mavsim_python
    - Chapter 11 assignment for Beard & McLain, PUP, 2012
    - Last Update:
        3/26/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM
import parameters.planner_parameters as PLAN

from chap3.data_viewer import dataViewer
from chap4.wind_simulation import windSimulation
from chap6.autopilot import autopilot
from chap7.mav_dynamics import mavDynamics
from chap8.observer import observer
from chap10.path_follower import path_follower
from chap11.path_manager import path_manager
from chap11.waypoint_viewer import waypoint_viewer

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
waypoint_view = waypoint_viewer()  # initialize the viewer
data_view = dataViewer()  # initialize view of data plots
if VIDEO == True:
    from chap2.video_writer import videoWriter
    video = video_writer(video_name="chap11_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
wind = windSimulation(SIM.ts_simulation)
mav = mavDynamics(SIM.ts_simulation)
ctrl = autopilot(SIM.ts_simulation)
obsv = observer(SIM.ts_simulation)
path_follow = path_follower()
path_manage = path_manager()

# waypoint definition
from message_types.msg_map import msgMap
from message_types.msg_waypoints import msgWaypoints
map_ = msgMap()
waypoints = msgWaypoints()
#waypoints.type = 'straight_line'
waypoints.type = 'fillet'
#waypoints.type = 'dubins'
Va = PLAN.Va0
waypoints.ned = np.array(
                [[0, 0, -100],
                [-1000, 700, -100],
                [0, 300, -100],
                [1000, 700, -100],
                [1500,1000, -100],
                [2000, 1500, -100],
                [0, 1000, -100],
                [1000, 1000, -100]]).T
waypoints.num_waypoints = waypoints.ned.shape[1]

waypoints.airspeed = np.array(
    [[Va, Va, Va, Va, Va, Va, Va, Va]])
waypoints.course = np.array(
    [[np.radians(0),
    np.radians(45),
    np.radians(45),
    np.radians(-135),
    np.radians(0),
    np.radians(45),
    np.radians(45),
    np.radians(-135)]])

# initialize the simulation time
sim_time = SIM.start_time
plot_time = 0
init = False 
# main simulation loop
print("Press Command-Q to exit...")
while sim_time < SIM.end_time:
    #-------observer-------------
    measurements = mav.sensors()  # get sensor measurements
    estimated_state = obsv.update(measurements)  # estimate states from measurements

    #-------path manager-------------
    path = path_manage.update(waypoints, PLAN.R_min, estimated_state)

    #-------path follower-------------
    autopilot_commands = path_follow.update(path, estimated_state)

    #-------controller-------------
    delta, commanded_state = ctrl.update(autopilot_commands, estimated_state)

    #-------physical system-------------
    current_wind = wind.update()  # get the new wind vector
    mav.update(delta, current_wind)  # propagate the MAV dynamics

    #-------update viewer-------------
    plot_time += SIM.ts_simulation
    if plot_time > SIM.ts_simulation*30 or path.flag_path_changed:
        waypoint_view.update(map_, waypoints, path, mav.true_state)  # plot path and MAV
        plot_time = 0
    data_view.update(mav.true_state, # true states
                    estimated_state, # estimated states
                    commanded_state, # commanded states
                    SIM.ts_simulation)
    if VIDEO == True: video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True: video.close()




