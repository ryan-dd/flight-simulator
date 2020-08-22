"""
msg_map
    - messages type for map of the world
    
part of mavsim_python
    - Beard & McLain, PUP, 2012
    - Last update:
        4/10/2019 - RWB
"""
import numpy as np
import parameters.planner_parameters as PLAN
from random import uniform
from shapely.geometry import Polygon, Point

class msgMap:
    def __init__(self):
        # flag to indicate if the map has changed
        self.flag_map_changed = 0
        # the city is of size (width)x(width)
        self.city_width = PLAN.city_width
        # number of blocks in city
        self.num_city_blocks = PLAN.num_blocks
        # percent of block that is street.
        self.street_width = PLAN.city_width / PLAN.num_blocks * PLAN.street_width
        # maximum height of buildings
        self.building_max_height = PLAN.building_height
        # an array of building heights
        self.building_height = PLAN.building_height * np.random.rand(PLAN.num_blocks, PLAN.num_blocks)
        # the width of the buildings (all the same)
        self.building_width = PLAN.city_width / PLAN.num_blocks * (1 - PLAN.street_width)
        # north coordinate of center of buildings
        self.building_north = np.zeros((1,PLAN.num_blocks))
        for i in range(PLAN.num_blocks):
            self.building_north[0, i] = 0.5 * (PLAN.city_width / PLAN.num_blocks) * (2 * i + 1)
        # east coordinate of center of buildings
        self.building_east = np.copy(self.building_north)

class msgMyMap:
    def __init__(self, num_buildings, bounds):
        self.buildings = [self.make_building(bounds) for i in range(num_buildings)]
    
    def make_building(self, bounds):
        height = np.random.rand()*150+50
        width = np.random.rand()*500+200
        n = uniform(0+width/2, bounds-width/2)
        e = uniform(0+width/2, bounds-width/2)
        return Building(height, width, n, e)
    
class Building():
    def __init__(self, height, width, n, e):
        self.height = height
        self.n = n
        self.e = e
        self.width = width
        self.points = [[e + width / 2, n + width / 2], #NE 0
                [e + width / 2, n - width / 2],   #SE 1
                [e - width / 2, n - width / 2],   #SW 2
                [e - width / 2, n + width / 2]]   #NW 3
        # self.check_points = [[e + width / 2, n + width / 2], #NE 0
        #         [e + width / (1.5), n - width / (1.5)],   #SE 1
        #         [e - width / (1.5), n - width / (1.5)],   #SW 2
        #         [e - width / (1.5), n + width / (1.5)]]   #NW 3
        clearance_distance = 50
        self.check_points = [[e + width / 2, n + width / 2], #NE 0
                [e + width / (2), n - width / (1.5)],   #SE 1
                [e - width / (2), n - width / (1.5)],   #SW 2
                [e - width / (2), n + width / (1.5)]]   #NW 3
        # self.check_points = [
        #     [0,0],
        #     [0,0],
        #     [0,0],
        #     [0,0]
        #     ]
        self.check_points = np.array(self.check_points)
        self.check_points[0,0] += clearance_distance
        self.check_points[1,0] += clearance_distance
        self.check_points[2,0] -= clearance_distance
        self.check_points[3,0] -= clearance_distance
        self.check_points[0,1] += clearance_distance
        self.check_points[3,1] += clearance_distance
        self.check_points[1,1] -= clearance_distance
        self.check_points[2,1] -= clearance_distance
        self.poly = Polygon(self.points)
        self.check_poly = Polygon(self.check_points)
    
    def get_coords(self):
        return self.points
    
    def get_discretized_coords(self, number_per_line):
        points1 = self.get_line_disc_coords_in_between(self.points[0], self.points[1], number_per_line)
        points2 = self.get_line_disc_coords_in_between(self.points[0], self.points[3], number_per_line)
        points3 = self.get_line_disc_coords_in_between(self.points[1], self.points[2], number_per_line)
        points4 = self.get_line_disc_coords_in_between(self.points[2], self.points[3], number_per_line)
        points1.extend(points2)
        points1.extend(points3)
        points1.extend(points4)
        points1.extend(self.points)
        return points1
        
    
    def get_line_disc_coords_in_between(self, point1, point2, number):
        point2 = np.array(point2)
        point1 = np.array(point1)
        line_length = np.linalg.norm(point2-point1)
        line_direction = (point2-point1)/line_length
        distance = line_length/(number+1)
        all_points = []
        for i in range(number):
            new_point = point1 + line_direction*distance*(i+1)
            all_points.append(new_point)
        return all_points
            
        
    def point_intersection(self, point):
        return self.poly.contains(Point(point[0], point[1]))
        

if __name__ == "__main__":
    map_ = msgMyMap(3, 4)