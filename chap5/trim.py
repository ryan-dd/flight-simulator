"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/29/2018 - RWB
        2/24/2020 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.rotations import Euler2Quaternion

def compute_trim(mav, Va, gamma):
    # define initial state and input
    e0, e1, e2, e3 = Euler2Quaternion(0., gamma, 0.)
    state = mav._state
    state0 = np.array([[state[0][0]],  # pn
                       [state[1][0]],  # pe
                       [state[2][0]],  # pd
                       [Va],  # u
                       [0],  # v
                       [0],  # w
                       [e0],  # e0
                       [e1],  # e1
                       [e2],  # e2
                       [e3],  # e3
                       [0],  # p
                       [0],  # q
                       [0]   # r
                   ])
    delta0 = np.array([[0],  # delta_e
                       [0],  # delta_a
                       [0],  # delta_r
                       [0.5]  # delta_t
                       ])
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7],  # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9],  # e3=0
                                x[10],  # p=0  - angular rates should all be zero
                                x[11],  # q=0
                                x[12],  # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs

    res = minimize(trim_objective_fun, x0, method='SLSQP', args=(mav, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_state[6:10] = trim_state[6:10]/np.linalg.norm(trim_state[6:10])
    trim_input = np.array([res.x[13:17]]).T
    mav._state = mav._state[:,None]
    print('trim_state=', trim_state.T)
    print('trim_input=', trim_input.T)
    return trim_state, trim_input

# objective function to be minimized
def trim_objective_fun(x, mav, Va, gamma):
    state_vars = x[0:13]
    delta_vars = x[13:17]
    mav._state = state_vars#np.asarray(state_vars)[:,None]
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta_vars)
    x_dot_current = mav._derivatives(mav._state, forces_moments)
    x_dot_desired = np.vstack((0,0, -Va*np.sin(gamma),0,0,0,0,0,0,0,0,0,0))
    J = np.linalg.norm(x_dot_desired[2:13] - x_dot_current[2:13])**2
    return J

