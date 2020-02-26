"""
mavsim_python: drawing tools
    - Beard & McLain, PUP, 2012
    - Update history:
        2/20/2019 - RWB
        4/15/2019 - BGM
        2/24/2020 - RWB
"""

import numpy as np
import pyqtgraph.opengl as gl
from tools.rotations import Euler2Rotation
# from chap11.dubins_parameters import dubinsParameters

class drawMav():
    def __init__(self, state, window):
        """
        Draw the MAV.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_meshColors = self.get_points()

        mav_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        self.mav_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
        window.addItem(self.mav_body)  # add body to plot

    def update(self, state):
        mav_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        # draw MAV by resetting mesh using rotated and translated points
        self.mav_body.setMeshData(vertexes=mesh, vertexColors=self.mav_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1, points.shape[1]]))
        return translated_points

    def get_points(self):
        """"
            Points that define the mav, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        scale = 4
        fuse_h = 2/scale
        fuse_w = 2/scale
        fuse_l1 = 4/scale
        fuse_l2 = 2/scale
        wing_l = 3/scale
        wing_w = 8/scale
        fuse_l3 = 8/scale
        tail_h = 2/scale
        tailwing_w = 4/scale
        tailwing_l = 2/scale
        #points are in North East Down coordinates
        points = np.array([[fuse_l1, 0, 0],  # point 1
                           [fuse_l2, fuse_w/2, fuse_h/2],  # point 2
                           [fuse_l2, -fuse_w/2, fuse_h/2],  # point 3
                           [fuse_l2, -fuse_w/2, -fuse_h/2],  # point 4
                           [fuse_l2, fuse_w/2, -fuse_h/2],  # point 5
                           [-fuse_l3, 0, 0],  # point 6
                           [0, wing_w/2, 0],  # point 7
                           [-wing_l, wing_w/2, 0],  # point 8
                           [-wing_l, -wing_w/2, 0],  # point 9
                           [0, -wing_w/2, 0],  # point 10
                           [-fuse_l3+tailwing_l, tailwing_w/2, 0],  # point 11
                           [-fuse_l3, tailwing_w/2, 0],  # point 12
                           [-fuse_l3, -tailwing_w/2, 0],  # point 13
                           [-fuse_l3+tailwing_l, -tailwing_w/2, 0],  # point 14
                           [-fuse_l3+tailwing_l, 0, 0],  # point 15
                           [-fuse_l3, 0, -tail_h],  # point 16
                          ]).T
        # scale points for better rendering
        scale = 10
        points = scale * points

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((13, 3, 4), dtype=np.float32)
        meshColors[0] = yellow  # front
        meshColors[1] = yellow  # front
        meshColors[2] = yellow  # back
        meshColors[3] = yellow  # back
        meshColors[4] = blue  # right
        meshColors[5] = blue  # right
        meshColors[6] = blue  # left
        meshColors[7] = blue  # left
        meshColors[8] = green  # top
        meshColors[9] = green  # top
        meshColors[10] = green  # bottom
        meshColors[11] = green  # bottom
        meshColors[12] = yellow  # bottom
        return points, meshColors

    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points=points.T
        mesh = np.array([[points[0], points[1], points[2]],  # front
                         [points[0], points[2], points[3]],  # front
                         [points[0], points[3], points[4]],  # front
                         [points[0], points[4], points[1]],  # front
                         [points[5], points[1], points[2]],  # body
                         [points[5], points[2], points[3]],  # body
                         [points[5], points[3], points[4]],  # body
                         [points[5], points[4], points[1]],  # body
                         [points[6], points[7], points[9]],  # front wing
                         [points[8], points[7], points[9]],  # front wing
                         [points[10], points[11], points[12]],  # back wing
                         [points[10], points[12], points[13]],  # back wing
                         [points[14], points[15], points[5]],  # tail
                         ])
        return mesh


class drawPath():
    def __init__(self, path, color, window):
        self.color = color
        if path.type == 'line':
            scale = 1000
            points = self.straight_line_points(path, scale)
        elif path.type == 'orbit':
            points = self.orbit_points(path)
        path_color = np.tile(color, (points.shape[0], 1))
        self.path_plot_object =  gl.GLLinePlotItem(pos=points,
                                                   color=path_color,
                                                   width=2,
                                                   antialias=True,
                                                   mode='line_strip')
        window.addItem(self.path_plot_object)

    def update(self, path):
        if path.type == 'line':
            scale = 1000
            points = self.straight_line_points(path, scale)
        elif path.type == 'orbit':
            points = self.orbit_points(path)
        self.path_plot_object.setData(pos=points)

    def straight_line_points(self, path, scale):
        points = np.array([[path.line_origin.item(0),
                            path.line_origin.item(1),
                            path.line_origin.item(2)],
                           [path.line_origin.item(0) + scale * path.line_direction.item(0),
                            path.line_origin.item(1) + scale * path.line_direction.item(1),
                            path.line_origin.item(2) + scale * path.line_direction.item(2)]])
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        return points


    def orbit_points(self, path):
        N = 10
        theta = 0
        theta_list = [theta]
        while theta < 2*np.pi:
            theta += 0.1
            theta_list.append(theta)
        points = np.array([[path.orbit_center.item(0) + path.orbit_radius,
                            path.orbit_center.item(1),
                            path.orbit_center.item(2)]])
        for angle in theta_list:
            new_point = np.array([[path.orbit_center.item(0) + path.orbit_radius * np.cos(angle),
                                   path.orbit_center.item(1) + path.orbit_radius * np.sin(angle),
                                   path.orbit_center.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        return points


class drawWaypoints():
    def __init__(self, waypoints, radius, color, window):
        self.radius = radius
        self.color = color
        if waypoints.type=='straight_line' or waypoints.type=='fillet':
            points = self.straight_waypoint_points(waypoints)
        elif waypoints.type=='dubins':
            points = self.dubins_points(waypoints, self.radius, 0.1)
        waypoint_color = np.tile(color, (points.shape[0], 1))
        self.waypoint_plot_object = gl.GLLinePlotItem(pos=points,
                                                      color=waypoint_color,
                                                      width=2,
                                                      antialias=True,
                                                      mode='line_strip')
        window.addItem(self.waypoint_plot_object)

    def update(self, waypoints):
        if waypoints.type=='straight_line' or waypoints.type=='fillet':
            points = self.straight_waypoint_points(waypoints)
        elif waypoints.type=='dubins':
            points = self.dubins_points(waypoints, self.radius, 0.1)
        self.waypoint_plot_object.setData(pos=points)

    def straight_waypoint_points(self, waypoints):
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = R @ np.copy(waypoints.ned)
        return points.T

    def dubins_points(self, waypoints, radius, Del):
        # returns a list of points along the dubins path
        initialize_points = True
        dubins_path = dubinsParameters()
        for j in range(0, waypoints.num_waypoints-1):
            dubins_path.update(
                waypoints.ned[:, j:j+1],
                waypoints.course.item(j),
                waypoints.ned[:, j+1:j+2],
                waypoints.course.item(j+1),
                radius)

            # points along start circle
            th1 = np.arctan2(dubins_path.p_s.item(1) - dubins_path.center_s.item(1),
                             dubins_path.p_s.item(0) - dubins_path.center_s.item(0))
            th1 = self.mod(th1)
            th2 = np.arctan2(dubins_path.r1.item(1) - dubins_path.center_s.item(1),
                             dubins_path.r1.item(0) - dubins_path.center_s.item(0))
            th2 = self.mod(th2)
            th = th1
            theta_list = [th]
            if dubins_path.dir_s > 0:
                if th1 >= th2:
                    while th < th2 + 2*np.pi:
                        th += Del
                        theta_list.append(th)
                else:
                    while th < th2:
                        th += Del
                        theta_list.append(th)
            else:
                if th1 <= th2:
                    while th > th2 - 2*np.pi:
                        th -= Del
                        theta_list.append(th)
                else:
                    while th > th2:
                        th -= Del
                        theta_list.append(th)

            if initialize_points:
                points = np.array([[dubins_path.center_s.item(0) + dubins_path.radius * np.cos(theta_list[0]),
                                    dubins_path.center_s.item(1) + dubins_path.radius * np.sin(theta_list[0]),
                                    dubins_path.center_s.item(2)]])
                initialize_points = False
            for angle in theta_list:
                new_point = np.array([[dubins_path.center_s.item(0) + dubins_path.radius * np.cos(angle),
                                       dubins_path.center_s.item(1) + dubins_path.radius * np.sin(angle),
                                       dubins_path.center_s.item(2)]])
                points = np.concatenate((points, new_point), axis=0)

            # points along straight line
            sig = 0
            while sig <= 1:
                new_point = np.array([[(1 - sig) * dubins_path.r1.item(0) + sig * dubins_path.r2.item(0),
                                       (1 - sig) * dubins_path.r1.item(1) + sig * dubins_path.r2.item(1),
                                       (1 - sig) * dubins_path.r1.item(2) + sig * dubins_path.r2.item(2)]])
                points = np.concatenate((points, new_point), axis=0)
                sig += Del

            # points along end circle
            th2 = np.arctan2(dubins_path.p_e.item(1) - dubins_path.center_e.item(1),
                             dubins_path.p_e.item(0) - dubins_path.center_e.item(0))
            th2 = self.mod(th2)
            th1 = np.arctan2(dubins_path.r2.item(1) - dubins_path.center_e.item(1),
                             dubins_path.r2.item(0) - dubins_path.center_e.item(0))
            th1 = self.mod(th1)
            th = th1
            theta_list = [th]
            if dubins_path.dir_e > 0:
                if th1 >= th2:
                    while th < th2 + 2 * np.pi:
                        th += Del
                        theta_list.append(th)
                else:
                    while th < th2:
                        th += Del
                        theta_list.append(th)
            else:
                if th1 <= th2:
                    while th > th2 - 2 * np.pi:
                        th -= Del
                        theta_list.append(th)
                else:
                    while th > th2:
                        th -= Del
                        theta_list.append(th)
            for angle in theta_list:
                new_point = np.array([[dubins_path.center_e.item(0) + dubins_path.radius * np.cos(angle),
                                       dubins_path.center_e.item(1) + dubins_path.radius * np.sin(angle),
                                       dubins_path.center_e.item(2)]])
                points = np.concatenate((points, new_point), axis=0)

        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        return points

    def mod(self, x):
        # force x to be between 0 and 2*pi
        while x < 0:
            x += 2*np.pi
        while x > 2*np.pi:
            x -= 2*np.pi
        return x

class drawMap():
    def __init__(self, map, window):
        self.window = window
        # draw map of the world: buildings
        fullMesh = np.array([], dtype=np.float32).reshape(0, 3, 3)
        fullMeshColors = np.array([], dtype=np.float32).reshape(0, 3, 4)
        for i in range(0, map.num_city_blocks):
            for j in range(0, map.num_city_blocks):
                mesh, meshColors = self.building_vert_face(map.building_north[0, i],
                                                           map.building_east[0, j],
                                                           map.building_width,
                                                           map.building_height[i, j])
                fullMesh = np.concatenate((fullMesh, mesh), axis=0)
                fullMeshColors = np.concatenate((fullMeshColors, meshColors), axis=0)
        self.ground_mesh = gl.GLMeshItem(
            vertexes=fullMesh,  # defines the triangular mesh (Nx3x3)
            vertexColors=fullMeshColors,  # defines mesh colors (Nx1)
            drawEdges=True,  # draw edges between mesh elements
            smooth=False,  # speeds up rendering
            computeNormals=False)  # speeds up rendering
        self.window.addItem(self.ground_mesh)

    def update(self, map):
        # draw map of the world: buildings
        fullMesh = np.array([], dtype=np.float32).reshape(0, 3, 3)
        fullMeshColors = np.array([], dtype=np.float32).reshape(0, 3, 4)
        for i in range(0, map.num_city_blocks):
            for j in range(0, map.num_city_blocks):
                mesh, meshColors = self.building_vert_face(map.building_north[0, i],
                                                           map.building_east[0, j],
                                                           map.building_width,
                                                           map.building_height[i, j])
                fullMesh = np.concatenate((fullMesh, mesh), axis=0)
                fullMeshColors = np.concatenate((fullMeshColors, meshColors), axis=0)
        self.ground_mesh.setData(vertexes=fullMesh, vertexColors=fullMeshColors)

    def building_vert_face(self, n, e, width, height):
        # define patches for a building located at (x, y)
        # vertices of the building
        points = np.array([[e + width / 2, n + width / 2, 0],  # NE 0
                           [e + width / 2, n - width / 2, 0],  # SE 1
                           [e - width / 2, n - width / 2, 0],  # SW 2
                           [e - width / 2, n + width / 2, 0],  # NW 3
                           [e + width / 2, n + width / 2, height],  # NE Higher 4
                           [e + width / 2, n - width / 2, height],  # SE Higher 5
                           [e - width / 2, n - width / 2, height],  # SW Higher 6
                           [e - width / 2, n + width / 2, height]])  # NW Higher 7
        mesh = np.array([[points[0], points[3], points[4]],  # North Wall
                         [points[7], points[3], points[4]],  # North Wall
                         [points[0], points[1], points[5]],  # East Wall
                         [points[0], points[4], points[5]],  # East Wall
                         [points[1], points[2], points[6]],  # South Wall
                         [points[1], points[5], points[6]],  # South Wall
                         [points[3], points[2], points[6]],  # West Wall
                         [points[3], points[7], points[6]],  # West Wall
                         [points[4], points[7], points[5]],  # Top
                         [points[7], points[5], points[6]]])  # Top

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        meshColors = np.empty((10, 3, 4), dtype=np.float32)
        meshColors[0] = green
        meshColors[1] = green
        meshColors[2] = green
        meshColors[3] = green
        meshColors[4] = green
        meshColors[5] = green
        meshColors[6] = green
        meshColors[7] = green
        meshColors[8] = yellow
        meshColors[9] = yellow
        return mesh, meshColors


