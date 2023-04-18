import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy
import warnings
from sklearn.gaussian_process.kernels import PairwiseKernel, Exponentiation, WhiteKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import polynomial_kernel, linear_kernel
import pandas as pd
from scipy.optimize import minimize
import torch
from sklearn import preprocessing
from scipy.stats import norm
from GP_param_train import GaussianProcessClassifierLaplace
from scipy.optimize._constraints import Bounds
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_environment')
from wrapper import BasicWrapper
sys.path.append('C:/Users/cvcla/my_py_projects/mf_PPO')
from constants import *
from model import Actor, Critic, MLPBase
from ppo import PPO
from running_state import *
from replay_memory import *


# make df to store X_a values
df = pd.DataFrame()
args = get_args()
env = BasicWrapper()
actor = Actor(env.observation_size, env.action_size, args.n_hidden)
critic = Critic(env.observation_size, args.n_hidden)  
MLPBase_model = MLPBase(env.observation_size, env.action_size, env.action_size) #what 3rd arg?
replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                observation_shape= env.observation_size,
                                action_dim=env.action_size)
running_state = ZFilter((env.observation_size,), clip=5)

ppo_agent = PPO(env, args, actor, critic, MLPBase_model) 

patients, S = env.reset() # S tensor
A = env.sample_random_action()
S_prime, R, pat, s_LogReg, r_LogReg, Xa, Xa_prime, outcome, done = env.step(A, S.detach().numpy())   




def sample_next_hyperparameter(acquisition_func, 
                               bounds=[[-10], [10]], n_restarts=25):
    """ sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[0], bounds[1], size=(n_restarts, n_params)):
        res = minimize(fun=acquisition_func,
                       x0=starting_point,
                       bounds=bounds,
                       method='L-BFGS-B')

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x

'''
Optimize method for continuous search space
include:
1. Exhausive search
2. Non-gradient method
3. Gradient ascent
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import torch

def MCSelector(func, model, mc_search_num = 1000):
    xspace = model.XspaceGenerate(mc_search_num)

    utilitymat = np.zeros(mc_search_num)+float('-Inf')

    if hasattr(model, 'multi_hyper') and model.multi_hyper:
            for i, x in enumerate(xspace):
                if hasattr(model, 'is_real_data') and model.is_real_data:
                    if i in model.dataidx:
                        continue
                x = xspace[i:i+1]
                for m in model.modelset:
                    utilitymat[i]+= func(x, m)
    else:
        for i, x in enumerate(xspace):
            if hasattr(model, 'is_real_data') and model.is_real_data:
                if i in model.dataidx:
                    continue
            x = xspace[i:i+1]# all the inputs should take 2d array 
            # if version == 'pytorch':
            #     x = torch.tensor(x, requires_grad=True)
            utilitymat[i] = func(x, model)
    
    max_value = np.max(utilitymat, axis = None)
    max_index = np.random.choice(np.flatnonzero(utilitymat == max_value))

    if hasattr(model, 'is_real_data') and model.is_real_data:
        model.dataidx = np.append(model.dataidx, max_index)

    # plt.figure()
    # plt.plot(xspace, utilitymat, 'ro')
    # plt.show()
    
    x = xspace[max_index]

    # plt.figure()
    # plt.plot(xspace, utilitymat)
    # plt.show()

    return x, max_value

def RandomSampling(model):
    x = model.XspaceGenerate(1)
    max_value = 0
    return x, max_value

def SGD(func, model, mc_search_num = 1000, learning_rate = 0.001):
    #for mm in range(100):
    random_num = round(0.7*mc_search_num)
    #x11, value11 = MCSelector(func, model, mc_search_num)
    x1, value1 = MCSelector(func, model, random_num)
    #x0 = model.XspaceGenerate(1).reshape(-1)
    x0 = torch.tensor(x1, requires_grad= True)
    optimizer = torch.optim.SGD([x0], lr=learning_rate)
    

    # for _ in range(round(0.3*mc_search_num)):
    #     loss = -func(x0, model, version='pytorch')

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     print("loss: {}".format(loss))
    
    # x0 = torch.tensor(x1, requires_grad= True)
    # optimizer = torch.optim.Adam([x0], lr=learning_rate)
    

    for _ in range(round(0.3*mc_search_num)):

        loss = -func(x0, model, version='pytorch')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("loss", loss)

    return x0.detach().numpy(), -loss

    # func2 = lambda x: -1.0*func(x, model)
    # bounds = np.array([model.xinterval[0], model.xinterval[1]])*np.ones((model.f_num, 2))
    # res = minimize(func2, x0, method='TNC', options={'disp':False}, bounds = bounds)
    # xstar = res.x
    # max_value = -res.fun
    # return xstar, max_value
    # max_value = float('-Inf')
    # for mm in range(50):
    #     x0 = model.XspaceGenerate(1).item()
    #     func2 = lambda x: -1.0*func(x, model)
    #     bounds = [(model.xinterval[0], model.xinterval[1])]
    #     res = minimize(func2, x0, method='TNC', 
    #                     options={ 'disp':False}, bounds = bounds)
    #     xstar22 = res.x
    #     max_value22 = -res.fun
    #     print(res)
    #     if max_value22.item() > max_value:
    #         max_value = max_value22.item()
    #         xstar = xstar22



    # # x0 = model.XspaceGenerate(1).item()
    # # func2 = lambda x: -1.0*func(x, model)
    # # bounds = [(-4, 4)]
    # # res = minimize(func2, x0, method='trust-constr', 
    # #                 options={#'xatol':1e-8, 
    # #                 'disp':True}, bounds = bounds)
    # # x = res.x
    # # max_value = -res.fun
    # return xstar, max_value
kernel = PairwiseKernel(metric = "polynomial")
#kernel = RBF() + WhiteKernel(noise_level=0.5)
#GPc = GaussianProcessClassifierLaplace(kernel = kernel)
#GPc.fit(preprocessing.normalize(Xa_prime.reshape(-1,1), norm='l2'), outcome.reshape(-1,1))
#print("theta", GPc.kernel_.theta[0])

#trial = GPc.fit_Xa()
#trial = sample_next_hyperparameter(GPc.f_x_i)

#print(GPc.kernel_(Xa_prime.reshape(-1,1), Xa.reshape(-1,1)))

def standardize_bounds(bounds, x0, meth):
    """Converts bounds to the form required by the solver."""
    if meth in {'trust-constr', 'powell', 'nelder-mead', 'new'}:
        
            lb, ub = old_bound_to_new(bounds)
            bounds = Bounds(lb, ub)
    elif meth in ('l-bfgs-b', 'tnc', 'slsqp', 'old'):
        
            bounds = new_bounds_to_old(bounds.lb, bounds.ub, x0.shape[0])
    return bounds
def _arr_to_scalar(x):
    # If x is a numpy array, return x.item().  This will
    # fail if the array has more than one element.
    return x.item() if isinstance(x, np.ndarray) else x
def old_bound_to_new(bounds):
    """Convert the old bounds representation to the new one.
    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are None they are replaced by
    -np.inf/np.inf.
    """
    lb = bounds[0]
    ub = bounds[1]


    # Convert occurrences of None to -inf or inf, and replace occurrences of
    # any numpy array x with x.item(). Then wrap the results in numpy arrays.
    lb = np.array([float(_arr_to_scalar(x)) if x is not None else -np.inf
                   for x in lb])
    ub = np.array([float(_arr_to_scalar(x)) if x is not None else np.inf
                   for x in ub])

    return lb, ub
def new_bounds_to_old(lb, ub, n):
    """Convert the new bounds representation to the old one.
    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are -np.inf/np.inf they are replaced by
    None.
    """
    lb = np.broadcast_to(lb, n)
    ub = np.broadcast_to(ub, n)

    lb = [float(x) if x > -np.inf else None for x in lb]
    ub = [float(x) if x < np.inf else None for x in ub]

    return list(zip(lb, ub))
'''
bounds_optim =np.array(([0, 10]))
value = [0, 10]
value_name = ['lb', 'ub']
n_params = bounds_optim.shape[0]

start_value = []
for starting_point in np.random.uniform(bounds_optim[0], bounds_optim[1], size=(25, n_params)):
     start_value.append(starting_point)
     #print("starting_point", starting_point)
print("bounds_optim", bounds_optim)
print("st value", start_value[0])
#print("stand bounds", standardize_bounds(list(zip(value_name, value )), start_value[0], "new"))
_b = [[0], [10]]


lb, ub = old_bound_to_new(_b)
print("lb ub", lb, ub)
bounds = Bounds(lb, ub)
print("bounds", bounds)
'''
     
#Xa_post = GPc.find_min() 
obj1 = np.inner(Xa_prime.reshape(-1, 1), Xa.reshape(-1,1)[0])  
obj2 = Xa_prime.T@obj1.T
der_tuple = [Xa_prime, 2*obj2.T]
print("der tuple", der_tuple)