import numpy as np
from scipy.spatial import distance_matrix
import parameters.planner_parameters as PLAN
from shapely.geometry import LineString, Polygon, Point
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, distance_matrix
from scipy.spatial.distance import pdist
import time

def calculate_voronoi_path(start_pos, end_pos, map_, buildings, bounds, plot=False):
    graph_points = construct_graph_points(buildings, bounds)
    connect_graph_points(graph_points, start_pos)

    
class Node():
    def __init__(self, position, goal):
        self.neighbors = []

def connect_graph_points(graph_points, start_pos, neighbors=5):
    graph_points = np.append(graph_points, np.array([start_pos]), axis=0)
    distances = distance_matrix(graph_points, graph_points)
    edge_list = np.argsort(distances)[:, 1:(neighbors+1)]

def construct_graph_points(buildings, bounds):
    fig, ax = plt.subplots()
    all_points = []
    vert = [[0,0],[0,bounds],[bounds,bounds],[bounds,0]]
    area = Polygon(vert)
    for building in buildings:
        points = building.get_discretized_coords(1)
        ax.plot(*building.poly.exterior.xy, color='b')
        # to_plot = np.array(points)
        # ax.scatter(to_plot[:,0], to_plot[:,1])
        all_points.extend(points)
    vor = Voronoi(all_points)
    points_for_graph = []
    for point in vor.vertices:
        intersects = check_if_point_intersects(point, area, buildings)
        if not intersects:
            points_for_graph.append(point)
    points_for_graph = np.array(points_for_graph)
    ax.scatter(points_for_graph[:,0], points_for_graph[:,1], color='r')
    plt.show()
    return points_for_graph

def check_if_point_intersects(point, outside_area, buildings):
    intersects = False
    if not outside_area.contains(Point(point[0], point[1])):
        return True
    for building in buildings:
        if building.point_intersection(point):
            intersects = True
            break
    return intersects


def check_if_line_intersects(new_configuration_pos, closest_pos, buildings):
    line = LineString([(new_configuration_pos.item(0), new_configuration_pos.item(1)),
                        (closest_pos.item(0), closest_pos.item(1))])
    intersects = False
    for building in buildings:
        if line.intersects(building.check_poly):
            intersects = True
            break
    return intersects