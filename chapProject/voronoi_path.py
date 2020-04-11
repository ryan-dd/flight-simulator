import numpy as np
from scipy.spatial import distance_matrix
import parameters.planner_parameters as PLAN
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

def calculate_voronoi_path(start_pos, end_pos, map_, buildings, plot=False):
    construct_voronoi_diagram(buildings)        

def construct_voronoi_diagram(buildings):
    fig, ax = plt.subplots()
    for building in buildings:
        points = building.get_discretized_coords(3)
        ax.plot(*building.poly.exterior.xy)
        to_plot = np.array(points)
        ax.scatter(to_plot[:,0], to_plot[:,1])
    plt.show()

