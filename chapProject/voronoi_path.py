import numpy as np
from scipy.spatial import distance_matrix
import parameters.planner_parameters as PLAN
from shapely.geometry import LineString, Polygon, Point
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, distance_matrix
from scipy.spatial.distance import pdist
import time

def calculate_voronoi_path(start_pos, end_pos, map_, buildings, bounds, plot=False):
    graph_points = construct_graph_points(buildings, bounds, plot=plot)
    return connect_graph_points(graph_points, start_pos, end_pos, buildings, plot=plot)

def connect_graph_points(graph_points, start_pos, end_pos, buildings, neighbors=100, plot=False):
    graph_points = np.append(np.array([start_pos]), graph_points, axis=0)
    graph_points = np.append(graph_points, np.array([end_pos]), axis=0)
    heuristic = distance_matrix(graph_points, np.array([end_pos]))
    distances = distance_matrix(graph_points, graph_points)
    edge_list = (np.argsort(distances)[:, 1:(neighbors+1)]).tolist()
    # Make sure start and end nodes are connected
    for edge in edge_list[-1]:
        edge_list[edge].append(len(edge_list)-1)
    for edge in edge_list[0]:
        edge_list[edge].append(len(edge_list)-1)
            
    g_values = np.zeros(len(graph_points))
    f_values = np.zeros(len(graph_points))
    parent = np.ones(len(graph_points))*-1
    open_list = set([0])
    closed_list = set([])
    curr_node_index = 0
    start = time.time()
    while len(open_list) != 0:
        open_list.discard(curr_node_index)
        closed_list.add(curr_node_index)
        if curr_node_index == len(graph_points)-1:
            # You win
            break
        for child_node_index in edge_list[curr_node_index]:
            if child_node_index in closed_list:
                continue
            if check_if_line_intersects(graph_points[child_node_index], graph_points[curr_node_index], buildings):
                continue
            if child_node_index in open_list:
                new_g = g_values[curr_node_index] + distances[curr_node_index][child_node_index]
                if new_g < g_values[child_node_index]:
                    g_values[child_node_index] = new_g
                    f_values[child_node_index] = g_values[child_node_index] + heuristic[child_node_index]
                    parent[child_node_index] = curr_node_index
            else:
                g_values[child_node_index] = g_values[curr_node_index] + distances[curr_node_index][child_node_index]
                f_values[child_node_index] = g_values[child_node_index] + heuristic[child_node_index]
                parent[child_node_index] = curr_node_index  
                open_list.add(child_node_index)
        list_open_list = list(open_list)
        f_values_open_list = np.array(f_values)[list(open_list)]
        curr_node_index = list_open_list[np.argmin(f_values_open_list)]
    final_path = []
    curr = len(graph_points)-1
    while curr != 0:
        final_path.append(graph_points[curr])
        curr = int(parent[curr])
    final_path.append(graph_points[curr])
    final_path.reverse()
    final_path = smooth_paths(final_path, buildings)
    if plot:
        to_plot = np.array(final_path)
        plt.plot(to_plot[:,0], to_plot[:,1])
    return np.array(final_path)
    
     
def smooth_paths(final_waypoints, buildings):
    waypoints = final_waypoints
    curr_index = 0
    next_index = 1
    all_waypoints = []
    all_waypoints.append(waypoints[curr_index])
    while True:
        waypoint = waypoints[curr_index]
        compare_waypoint = waypoints[next_index]
        intersects = check_if_line_intersects(compare_waypoint, waypoint, buildings)
        if not intersects:
            next_index += 1
        else:
            all_waypoints.append(waypoints[next_index-1])
            curr_index = next_index-1            
        if next_index == len(final_waypoints):
            all_waypoints.append(final_waypoints[next_index-1])
            break
    return all_waypoints

def construct_graph_points(buildings, bounds, plot=False):
    if plot:
        fig, ax = plt.subplots()
    all_points = []
    vert = [[0,0],[0,bounds],[bounds,bounds],[bounds,0]]
    area = Polygon(vert)
    for building in buildings:
        points = building.get_discretized_coords(1)
        if plot:
            ax.plot(*building.poly.exterior.xy, color='b')
            to_plot = np.array(points)
            # ax.scatter(to_plot[:,0], to_plot[:,1], color='b')
        all_points.extend(points)
    vor = Voronoi(all_points)
    points_for_graph = []
    for point in vor.vertices:
        intersects = check_if_point_intersects(point, area, buildings)
        if not intersects:
            points_for_graph.append(point)
    points_for_graph = np.array(points_for_graph)
    if plot:
        ax.scatter(points_for_graph[:,0], points_for_graph[:,1], color='r')
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


def check_if_line_intersects(next_pos, prev_pos, buildings):
    line = LineString([(next_pos.item(0), next_pos.item(1)),
                        (prev_pos.item(0), prev_pos.item(1))])
    intersects = False
    for building in buildings:
        if line.intersects(building.poly):
            intersects = True
            break
    return intersects