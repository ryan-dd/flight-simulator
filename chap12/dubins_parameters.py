# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB

import numpy as np
import sys
sys.path.append('..')


class dubins_parameters:
    def __init__(self):
        self.p_s = np.inf*np.ones((3,1))  # the start position in re^3
        self.chi_s = np.inf  # the start course angle
        self.p_e = np.inf*np.ones((3,1))  # the end position in re^3
        self.chi_e = np.inf  # the end course angle
        self.radius = np.inf  # turn radius
        self.length = np.inf  # length of the Dubins path
        self.center_s = np.inf*np.ones((3,1))  # center of the start circle
        self.dir_s = np.inf  # direction of the start circle
        self.center_e = np.inf*np.ones((3,1))  # center of the end circle
        self.dir_e = np.inf  # direction of the end circle
        self.r1 = np.inf*np.ones((3,1))  # vector in re^3 defining half plane H1
        self.r2 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H2
        self.r3 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H3
        self.n1 = np.inf*np.ones((3,1))  # unit vector in re^3 along straight line path
        self.n3 = np.inf*np.ones((3,1))  # unit vector defining direction of half plane H3

    def update(self, p_start, chi_start, p_end, chi_end, R):
        ell = np.linalg.norm(p_start - p_end)
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
        else:
            crs = p_start + R*rotz(np.pi/2) @ np.vstack((
            np.cos(chi_start), 
            np.sin(chi_start),
            0))
            cls_ = p_start + R*rotz(-np.pi/2) @ np.vstack((
            np.cos(chi_start), 
            np.sin(chi_start),
            0)) 
            cre = p_end + R*rotz(np.pi/2) @ np.vstack((
            np.cos(chi_end), 
            np.sin(chi_end),
            0))
            cle = p_end + R*rotz(-np.pi/2) @ np.vstack((
            np.cos(chi_end), 
            np.sin(chi_end),
            0)) 

            vee = np.arctan2(cre.item(1)-crs.item(1), cre.item(0)-crs.item(0))
            L1 = self.calculate_L1(crs, cre, R, vee, chi_start, chi_end)
            
            ell = np.linalg.norm(cle - crs)
            vee = np.arctan2(cle.item(1)-crs.item(1), cle.item(0)-crs.item(0))
            vee2 = vee - np.pi/2 + np.arcsin(2*R/ell)
            L2 = self.calculate_L2(ell, R, vee, vee2, chi_start, chi_end)
            
            ell = np.linalg.norm(cre-cls_)
            vee = np.arctan2(cre.item(1)-cls_.item(1), cre.item(0)-cls_.item(0))
            vee2 = np.arccos(2*R/ell)
            L3 = self.calculate_L3(ell, R, vee, vee2, chi_start, chi_end)
            
            vee = np.arctan2(cle.item(1)-cls_.item(1), cle.item(0)-cls_.item(0))
            L4 = self.calculate_L4(cls_, cle, R, vee, chi_start, chi_end)
            
            L = np.array([L1, L2, L3, L4])
            self.length = min(L)
            arg_min = np.argmin(L).item(0)+1
            e1 = np.array([[1], [0], [0]])
            if arg_min == 1:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cre
                self.dir_e = 1
                q1 = (self.center_e - self.center_s)/np.linalg.norm((self.center_e - self.center_s))
                z1 = self.center_s + R*rotz(-np.pi/2) @ q1
                z2 = self.center_e + R*rotz(-np.pi/2) @ q1
                self.n1 = q1
                self.r1 = z1
                self.r2 = z2
            elif arg_min == 4:
                self.center_s = cls_
                self.dir_s = -1
                self.center_e = cle
                self.dir_e = -1
                q1 = (self.center_e - self.center_s)/np.linalg.norm((self.center_e - self.center_s))
                z1 = self.center_s + R*rotz(np.pi/2) @ q1
                z2 = self.center_e + R*rotz(np.pi/2) @ q1
                self.n1 = q1
                self.r1 = z1
                self.r2 = z2
            elif arg_min == 2:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cle
                self.dir_e = -1
                ell = np.linalg.norm(cle-crs)
                vee = np.arctan2(cle.item(1)-crs.item(1), cle.item(0)-crs.item(0))
                vee2 = vee - np.pi/2 + np.arcsin(2*R/ell)
                q1 = rotz(vee2 + np.pi/2) @ e1
                z1 = self.center_s + R*rotz(vee2) @ e1
                z2 = self.center_e + R*rotz(vee2 + np.pi) @ e1
                self.n1 = q1
                self.r1 = z1
                self.r2 = z2
            elif arg_min == 3:
                self.center_s = cls_
                self.dir_s = -1
                self.center_e = cre
                self.dir_e = 1
                ell = np.linalg.norm(cre-cls_)
                vee = np.arctan2(cre.item(1)-cls_.item(1), cre.item(0)-cls_.item(0))
                vee2 = np.arccos(2*R/ell)
                q1 = rotz(vee + vee2 - np.pi/2) @ e1
                z1 = self.center_s + R*rotz(vee + vee2) @ e1
                z2 = self.center_e + R*rotz(vee + vee2 - np.pi) @ e1
                self.n1 = q1
                self.r1 = z1
                self.r2 = z2
            self.radius = R
            self.p_s = p_start
            self.chi_s = chi_start
            self.p_e = p_end
            self.chi_e = chi_end  
            self.r3 = p_end
            self.n3 = rotz(chi_end) @ e1
                    
    def calculate_L1(self, crs, cre, R, vee, chi_start, chi_end):
        length_straight_line = np.linalg.norm(crs - cre)
        length_crs = R*mod(2*np.pi + mod(vee - np.pi/2) - mod(chi_start - np.pi/2))
        length_cre = R*mod(2*np.pi + mod(chi_end - np.pi/2) - mod(vee - np.pi/2))
        L1 = length_straight_line + length_crs + length_cre
        return L1

    def calculate_L2(self, ell, R, vee, vee2, chi_start, chi_end):
        length_straight_line = np.sqrt(ell**2 - 4*R**2)
        length_crs = R*mod(2*np.pi + mod(vee2) - mod(chi_start - np.pi/2))
        length_cle = R*mod(2*np.pi + mod(vee2 + np.pi) - mod(chi_end + np.pi/2))
        L2 = length_straight_line + length_crs + length_cle
        return L2
    
    def calculate_L3(self, ell, R, vee, vee3, chi_start, chi_end):
        length_straight_line = np.sqrt(ell**2 - 4*R**2)
        length_cls = R*mod(2*np.pi + mod(chi_start + np.pi/2) - mod(vee + vee3))
        length_cre = R*mod(2*np.pi + mod(chi_end - np.pi/2) - mod(vee + vee3 - np.pi))
        L3 = length_straight_line + length_cls + length_cre
        return L3
    
    def calculate_L4(self, cls_, cle, R, vee, chi_start, chi_end):
        length_straight_line = np.linalg.norm(cls_-cle)
        length_cls = R*mod(2*np.pi + mod(chi_start + np.pi/2) - mod(vee + np.pi/2))
        length_cle = R*mod(2*np.pi + mod(vee + np.pi/2) - (chi_end + np.pi/2))
        L4 = length_straight_line + length_cls + length_cle
        return L4
    
def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def mod(x):
    # make x between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x


