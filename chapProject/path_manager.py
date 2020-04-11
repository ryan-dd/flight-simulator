import numpy as np
import sys
sys.path.append('..')
from chap12RRT.dubins_parameters import dubins_parameters
from message_types.msg_path import msgPath

class path_manager:
    def __init__(self):
        # message sent to path follower
        self.path = msgPath()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        # flag that request new waypoints from path planner
        self.flag_need_new_waypoints = True
        self.delay = False
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        # dubins path parameters
        self.dubins_path = dubins_parameters()

    def update(self, waypoints, radius, state):
        # this flag is set for one time step to signal a redraw in the viewer
        if self.path.flag_path_changed:
            self.path.flag_path_changed = False
            self.delay = False
        if self.delay:
            self.path.flag_path_changed = True
        if waypoints.num_waypoints == 0:
            waypoints.flag_manager_requests_waypoints = True
        else:
            if waypoints.type == 'straight_line':
                self.line_manager(waypoints, state)
            elif waypoints.type == 'fillet':
                self.fillet_manager(waypoints, radius, state)
            elif waypoints.type == 'dubins':
                self.dubins_manager(waypoints, radius, state)
            else:
                print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        p = np.array([[state.pn], [state.pe], [-state.h]])
        w_prev = (waypoints.ned[:, self.ptr_previous]).reshape(-1,1)
        w_curr = (waypoints.ned[:, self.ptr_current]).reshape(-1,1)
        w_next = (waypoints.ned[:, self.ptr_next]).reshape(-1,1)
        vector_diff_prev = w_curr - w_prev
        q_prev = vector_diff_prev/np.linalg.norm(vector_diff_prev)
        vector_diff_next = w_next - w_curr
        q_curr = vector_diff_next/np.linalg.norm(vector_diff_next)
        self.halfspace_r = w_curr
        self.halfspace_n = q_curr + q_prev/(np.linalg.norm(q_curr + q_prev))
        if self.inHalfSpace(p):
            self.increment_pointers(waypoints)
            self.delay = True
        self.path.line_origin = w_prev
        self.path.line_direction = q_prev
            

    def fillet_manager(self, waypoints, radius, state):
        p = np.array([[state.pn], [state.pe], [-state.h]])
        w_prev = (waypoints.ned[:, self.ptr_previous]).reshape(-1,1)
        w_curr = (waypoints.ned[:, self.ptr_current]).reshape(-1,1)
        w_next = (waypoints.ned[:, self.ptr_next]).reshape(-1,1)
        vector_diff_prev = w_curr - w_prev
        q_prev = vector_diff_prev/np.linalg.norm(vector_diff_prev)
        vector_diff_next = w_next - w_curr
        q_curr = vector_diff_next/np.linalg.norm(vector_diff_next)
        psi = np.arccos(-q_prev.T @ q_curr)
        if self.manager_state == 1:
            self.path.type = 'line'
            self.path.line_origin = w_prev
            self.path.line_direction = q_prev
            self.halfspace_r = w_curr-(radius/np.tan(psi/2)) * q_prev
            self.halfspace_n = q_prev
            if self.inHalfSpace(p):
                self.manager_state = 2
                self.delay = True
        elif self.manager_state == 2:
            self.path.type = 'orbit'
            norm_diff = (q_prev - q_curr)/np.linalg.norm(q_prev-q_curr)
            self.path.orbit_center = w_curr - radius/np.sin(psi/2) * norm_diff
            self.path.orbit_radius = radius
            self.path.orbit_direction = self.orbit_direction_string(np.sign(q_prev.item(0)*q_curr.item(1) - q_prev.item(1)*q_curr.item(0)))
            self.halfspace_r = w_curr + (radius/np.tan(psi/2)) * q_curr
            self.halfspace_n = q_curr
            if self.inHalfSpace(p):
                self.manager_state = 1
                self.increment_pointers(waypoints)
                self.delay = True
        # self.path.line_origin = w_prev
        # self.path.line_direction = q_prev

    def dubins_manager(self, waypoints, radius, state):
        p = np.array([[state.pn], [state.pe], [-state.h]])
        w_prev = (waypoints.ned[:, self.ptr_previous]).reshape(-1,1)
        chi_prev = waypoints.course.item(self.ptr_previous)
        w_curr = (waypoints.ned[:, self.ptr_current]).reshape(-1,1)
        chi_curr = waypoints.course.item(self.ptr_current)
        self.dubins_path.update(w_prev, chi_prev, w_curr, chi_curr, radius)
        if self.manager_state == 1:
            self.path.type = 'orbit'
            self.path.orbit_center = self.dubins_path.center_s
            self.path.orbit_radius = self.dubins_path.radius
            self.path.orbit_direction = self.orbit_direction_string(self.dubins_path.dir_s)
            self.halfspace_r = self.dubins_path.r1
            self.halfspace_n = -self.dubins_path.n1
            if self.inHalfSpace(p):
                self.manager_state = 2
                self.delay = True
        elif self.manager_state == 2:
            self.halfspace_r = self.dubins_path.r1
            self.halfspace_n = self.dubins_path.n1
            if self.inHalfSpace(p):
                self.manager_state = 3
                self.delay = True
        elif self.manager_state == 3:
            self.path.type = 'line'
            self.path.line_origin = self.dubins_path.r1
            self.path.line_direction = self.dubins_path.n1
            self.halfspace_r = self.dubins_path.r2
            self.halfspace_n = self.dubins_path.n1
            if self.inHalfSpace(p):
                self.manager_state = 4
                self.delay = True
        elif self.manager_state == 4:
            self.path.type = 'orbit'
            self.path.orbit_center = self.dubins_path.center_e
            self.path.orbit_radius = self.dubins_path.radius
            self.path.orbit_direction = self.orbit_direction_string(self.dubins_path.dir_e)
            self.halfspace_r = self.dubins_path.r3
            self.halfspace_n = -self.dubins_path.n3
            if self.inHalfSpace(p):
                self.manager_state = 5
                self.delay = True
        elif self.manager_state == 5:
            self.halfspace_n = self.dubins_path.n3
            if self.inHalfSpace(p):
                self.manager_state = 1
                self.increment_pointers(waypoints)
                self.delay = True
        

    def initialize_pointers(self):
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2

    def increment_pointers(self, waypoints):
        self.ptr_previous += 1
        self.ptr_current += 1
        self.ptr_next += 1
        if self.ptr_previous == waypoints.num_waypoints:
            self.ptr_previous = 0
        if self.ptr_current == waypoints.num_waypoints:
            self.ptr_current = 0
        if self.ptr_next == waypoints.num_waypoints:
            self.ptr_next = 0

    def inHalfSpace(self, pos):
        if (pos - self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

    def orbit_direction_string(self, value):
        if value == 1:
            res = 'CW'
        elif value == -1:
            res = 'CCW'
        return res