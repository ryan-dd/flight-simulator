# path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/3/2019 - BGM
import numpy as np
import sys
sys.path.append('..')
from message_types.msg_waypoints import msgWaypoints

class path_planner:
    def __init__(self):
        # waypoints definition
        self.waypoints = msgWaypoints()

    def update(self, map, state):
        # this flag is set for one time step to signal a redraw in the viewer
        #planner_flag = 1  # return simple waypoint path
        planner_flag = 2  # return dubins waypoint path
        # planner_flag = 3  # plan path through city using straight-line RRT
        # planner_flag = 4  # plan path through city using dubins RRT
        if planner_flag == 1:
            self.waypoints.type = 'fillet'
            self.waypoints.num_waypoints = 4
            Va = 25
            self.waypoints.ned = np.array([[0, 0, -100],
                            [1000, 0, -100],
                            [0, 1000, -100],
                            [1000, 1000, -100]]).T
            self.waypoints.airspeed = np.array([[Va, Va, Va, Va]])
        elif planner_flag == 2:
            self.waypoints.type = 'dubins'
            self.waypoints.num_waypoints = 4
            Va = 25
            self.waypoints.ned = np.array(
                [[0, 0, -100],
                [-700, 200, -100],
                [0, -400, -100],
                [1000, 700, -100]]).T
            self.waypoints.airspeed = np.array(
                [[Va, Va, Va, Va]])
            self.waypoints.course = np.array(
                [[np.radians(0),
                np.radians(45),
                np.radians(45),
                np.radians(-135)]])
        elif planner_flag == 3:
            self.waypoints.type = 'fillet'
            self.waypoints.num_waypoints = 0
            Va = 25
            # current configuration vector format: N, E, D, Va
            wpp_start = np.array([state.n,
                                  state.e,
                                  -state.h,
                                  state.Va])
            if np.linalg.norm(np.array([state.n, state.e, -state.h])-np.array([map.city_width, map.city_width, -state.h])) == 0:
                wpp_end = np.array([0,
                                    0,
                                    -state.h,
                                    Va])
            else:
                wpp_end = np.array([map.city_width,
                                    map.city_width,
                                    -state.h,
                                    Va])

            waypoints = self.rrt.planPath(wpp_start, wpp_end, map)
            self.waypoints.ned = waypoints.ned
            self.waypoints.airspeed = waypoints.airspeed
            self.waypoints.num_waypoints = waypoints.num_waypoints
        # elif planner_flag == 4:

        else:
            print("Error in Path Planner: Undefined planner type.")

        return self.waypoints
