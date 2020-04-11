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
from random import random
from shapely.geometry import Polygon

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
        self.all_buildings = [self.make_building() for i in range(num_buildings)]
    
    def make_building(self):
        height = np.random.rand()*150+50
        n = np.random.rand(0+bounds/2, bounds-bounds/2)
        e = np.random.rand(0+bounds/2, bounds-bounds/2)
        width = 30
        return Building(height, n, e, width)
    
class Building():
    def __init__(self, height, width, n, e):
        self.height = height
        self.n = n
        self.e = e
        self.points = [[e + width / 2, n + width / 2], #NE 0
                [e + width / 2, n - width / 2],   #SE 1
                [e - width / 2, n - width / 2],   #SW 2
                [e - width / 2, n + width / 2]]   #NW 3
        self.check_points = [[e + width / 2, n + width / 2], #NE 0
                [e + width / (1.5), n - width / (1.5)],   #SE 1
                [e - width / (1.5), n - width / (1.5)],   #SW 2
                [e - width / (1.5), n + width / (1.5)]]   #NW 3
        self.poly = Polygon(self.points)
        self.check_poly = Polygon(self.check_points)
    
    def get_coords(self):
        return self.points

    def point_intersection(self, point):
        return self.poly.contains(Point(point.position[0], point.position[1]))
        

if __name__ == "__main__":
    map_ = msgMyMap(3, 4)