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
from chapProject.path_manager import path_manager
from chapProject.waypoint_viewer import waypoint_viewer
from chapProject.voronoi_path import calculate_voronoi_path

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
from message_types.msg_map import msgMyMap
from message_types.msg_waypoints import msgWaypoints
bounds = 2000
map_ = msgMyMap(10, bounds)
waypoints = msgWaypoints()
#waypoints.type = 'straight_line'
waypoints.type = 'fillet'
#waypoints.type = 'dubins'
Va = PLAN.Va0
start_node = [0, 0]
end_node = [bounds, bounds]
buildings = waypoint_view.drawMap(map_)
waypoints.ned = calculate_voronoi_path(start_node, end_node, map_, map_.buildings)
waypoints.num_waypoints = waypoints.ned.shape[0]
waypoints.ned = np.append(waypoints.ned, np.ones((waypoints.ned.shape[0], 1))*-100, axis=1).T
Va = PLAN.Va0
waypoints.airspeed = np.ones(waypoints.num_waypoints)*Va
all_angles = []
for i, waypoint in enumerate(waypoints.ned.T):
    if i == 0:
        prev = waypoint
    else: 
        direction = (waypoint[0:2] - prev[0:2])/np.linalg.norm(waypoint[0:2])
        angle = np.arctan2(direction.item(1), direction.item(0))
        prev = waypoint
        all_angles.append(angle)
all_angles.append(angle)
waypoints.course = np.array(all_angles)

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
    if plot_time > SIM.ts_simulation*30 or path.flag_path_changed or not init:
        waypoint_view.update(map_, waypoints, path, mav.true_state)  # plot path and MAV
        plot_time = 0
        init=True
    data_view.update(mav.true_state, # true states
                    estimated_state, # estimated states
                    commanded_state, # commanded states
                    SIM.ts_simulation)
    if VIDEO == True: video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True: video.close()




