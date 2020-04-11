import numpy as np
from scipy.spatial import distance_matrix
import parameters.planner_parameters as PLAN
from shapely.geometry import LineString
import matplotlib.pyplot as plt

def calculate_RRT_path(start_pos, end_pos, map_, buildings, plot=False):   
    node_container = NodeContainer(start_pos)
    D = 550
    if plot:
        fig, ax = plt.subplots()
        for building in buildings:
            ax.plot(*building.poly.exterior.xy)
    while True:
        new_position = randomly_select_new_node(PLAN.city_width)
        closest_node = find_closest_node(new_position, 
                                         node_container.all_nodes_position_list, 
                                         node_container.all_nodes_list)
        new_configuration_position = get_new_configuration(new_position, closest_node.coords, D)
        intersects = check_if_intersects(new_configuration_position, closest_node.coords, buildings)
        if not intersects:
            node_container.insert_node(new_configuration_position, closest_node)
            if plot:
                to_plot = np.vstack((new_configuration_position, closest_node.coords))
                ax.plot(to_plot[:,0], to_plot[:,1])
            if point_distance(new_configuration_position, end_pos) < D:
                node_container.insert_node(np.array([end_pos]), node_container.get_last_node())
                if plot:
                    to_plot = np.vstack((new_configuration_position, np.array(end_pos)))
                    ax.plot(to_plot[:,0], to_plot[:,1])
                    plt.show()
                break
    final_waypoints = []
    final_waypoints.append(np.array([end_pos]))
    parent = node_container.get_last_node().parent
    while parent is not None:
        final_waypoints.append(parent.coords)
        parent = parent.parent
    hi = 0
    final_waypoints.reverse()
    final_points = smooth_paths(final_waypoints, buildings)
    return np.array(final_waypoints).reshape(-1,2)
         
def smooth_paths(final_waypoints, buildings):
    waypoints = final_waypoints
    curr_index = 0
    next_index = 1
    all_waypoints = []
    all_waypoints.append(waypoints[curr_index])
    while True:
        waypoint = waypoints[curr_index]
        compare_waypoint = waypoints[next_index]
        intersects = check_if_intersects(compare_waypoint, waypoint, buildings)
        if not intersects:
            next_index += 1
        else:
            all_waypoints.append(waypoints[next_index-1])
            curr_index = next_index-1            
        if next_index == len(final_waypoints):
            all_waypoints.append(final_waypoints[next_index-1])
            break
    return all_waypoints
        
            
def randomly_select_new_node(city_width):
    return np.random.rand(1,2)*PLAN.city_width

def find_closest_node(new_position, all_nodes_positions, all_nodes):
    distances = distance_matrix(new_position, all_nodes_positions)
    return all_nodes[np.argmin(distances)]

def get_new_configuration(new_position, closest_node, D):
    direction = (new_position - closest_node) / np.linalg.norm(new_position - closest_node)
    return closest_node + direction*D

def check_if_intersects(new_configuration_pos, closest_pos, buildings):
    line = LineString([(new_configuration_pos.item(0), new_configuration_pos.item(1)),
                        (closest_pos.item(0), closest_pos.item(1))])
    intersects = False
    for building in buildings:
        if line.intersects(building.check_poly):
            intersects = True
            break
    return intersects


def point_distance(pos1, pos2):
    return np.linalg.norm(pos1-pos2)

class NodeContainer():
    def __init__(self, start_position):
        start_node = Node(np.array([start_position]), None)
        self.all_nodes_position_list = np.array([start_position])
        self.all_nodes_list = [start_node]
        
    def insert_node(self, position, parent_node):
        next_node = Node(position, parent_node)
        # self.node_dict[next_node.key()] = next_node
        self.all_nodes_position_list = np.append(self.all_nodes_position_list, position, axis=0)
        self.all_nodes_list.append(next_node)
    
    def get_last_node(self):
        return self.all_nodes_list[-1]
        
class Node():
    def __init__(self, coords, parent):
        self.coords = coords
        self.children = []
        self.parent = parent
        if parent is not None:
            parent.add_child(self)
    
    def key(self):
        return str(coords.item(0)) + " " + str(coords.item(1))
    
    def add_child(self, child):
        self.children.append(child)
