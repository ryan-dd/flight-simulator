"""
Class to determine wind velocity at any given moment,
calculates a steady wind speed and uses a stochastic
process to represent wind gusts. (Follows section 4.4 in uav book)
"""
import sys
sys.path.append('..')
from tools.transfer_function import TransferFunction
import numpy as np

class wind_simulation:
    def __init__(self, Ts):
        # steady state wind defined in the inertial frame
        self._steady_state = np.array([[5],[0],[-5]])
        Va = 25
        Lu = 200
        Lv = 200
        Lw = 50
        include_gust = True 
        if include_gust:
            sigma_u = 1.06
            sigma_v = 1.06
            sigma_w = 0.7

        a1 = sigma_u*np.sqrt(Va/Lu)
        b1 = Va/Lu
        
        a2 = sigma_v*np.sqrt(3*Va/Lv)
        a3 = a2*Va/(np.sqrt(3)*Lv)
        b2 = Va/Lv

        a4 = sigma_w*np.sqrt(3*Va/Lw)
        a5 = a4*Va/(np.sqrt(3)*Lw)
        b3 = Va/Lw
        self.u_w = TransferFunction(num=np.array([[a1]]),
                                     den=np.array([[1, b1]]),
                                     Ts=Ts)
        self.v_w = TransferFunction(num=np.array([[a2, a3]]),
                                     den=np.array([[1, 2*b2, b2**2.0]]),
                                     Ts=Ts)
        self.w_w = TransferFunction(num=np.array([[a4, a5]]),
                                     den=np.array([[1, 2*b3, b3**2.0]]),
                                     Ts=Ts)
        self._Ts = Ts

    def update(self):
        # returns a six vector.
        #   The first three elements are the steady state wind in the inertial frame
        #   The second three elements are the gust in the body frame
        gust = np.array([[self.u_w.update(np.random.randn())],
                         [self.v_w.update(np.random.randn())],
                         [self.w_w.update(np.random.randn())]])
        # gust = np.array([[self.u_w.update(0)],
        #                  [self.v_w.update(0)],
        #                  [self.w_w.update(0)]])
        # gust = np.array([[0.],[0.],[0.]])
        return np.concatenate(( self._steady_state, gust ))
