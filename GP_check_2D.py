import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.functional import F
from copy import copy
import time
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
default_seed = 10000
from sklearn.gaussian_process.kernels import PairwiseKernel, Exponentiation, WhiteKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit as sigmoid
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import polynomial_kernel, linear_kernel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from scipy.optimize import minimize
import torch
from sklearn import preprocessing
from scipy.stats import norm
from scipy.optimize._constraints import Bounds
import seaborn as sns
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_environment')
from wrapper import BasicWrapper
sys.path.append('C:/Users/cvcla/my_py_projects/mf_PPO')
from constants import *
from model import Actor, Critic, MLPBase
from ppo import PPO
from running_state import *
from replay_memory import *
from utilis import *


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

patients, S, Y, count = env.reset() # S tensor
A = env.sample_random_action()
output = env.multi_step(A.detach().numpy(), S, Y, patients) 
df =np.vstack([output["patients"][0][:, 2], output["Xa_post"][0]]).reshape(-1, 2) # pat is 1, Xs, Xaprint("classes", np.sum(outcome)) 
kernel = PairwiseKernel(metric = "polynomial", pairwise_kernels_kwargs={"degree":2, "coef0":0})
print("Ply kernel", kernel)
#kernel = RBF() + WhiteKernel(noise_level=0.5)
#df = pd.DataFrame(data={'Xs':pat[:, 1], 'Xa': Xa_prime}).to_numpy()
XS = np.random.normal(0,1, size=(200,1))
print("prescal" , XS.shape,output["Xa_post"][0].shape )
pre_scaling = np.column_stack([XS, output["Xa_post"][0]]).reshape(-1, XS.shape[1]+1) # this df potentially can grow in dim 1
scaler = preprocessing.StandardScaler().fit(df)
X = scaler.transform(df)

y = Y

train_size = 100
X_2d_train = X[:train_size, 0:2]; y_2d_train = y[:train_size, ]; X_2d_test = X[:train_size, 0:2]; y_2d_test = y[train_size:, ]
print("Data")
print("data shape - np array - Xa Xs", X.shape[0], y.shape, type(X))
print("TRAIN data - x train", X_2d_train.shape ,"y train", y_2d_train.shape)
print("TEST data - x train", X_2d_test.shape ,"y train", y_2d_test.shape)

X_torch = torch.from_numpy(X).float();y_torch = torch.from_numpy(y).float()
X_train_torch = torch.from_numpy(X_2d_train).float(); y_train_torch = torch.from_numpy(y_2d_train).float(); X_test_torch = torch.from_numpy(X_2d_test).float(); y_test_torch = torch.from_numpy(y_2d_test).float()

print("data shape - torch", X_torch.shape, y_torch.shape, type(X_torch), X_torch.size(0))
print("TRAIN data - x train", X_train_torch.shape ,"y train", y_train_torch.shape)
print("TEST data - x train", X_test_torch.shape ,"y train", y_test_torch.shape)


fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained"
)
sns.scatterplot(x=X_2d_train[:, 0], y=X_2d_train[:, 1], s=50, hue=y_2d_train, ax=ax[0])
ax[0].set(title="Raw Data - Training Set")
sns.scatterplot(x=X_2d_test[:, 0], y=X_2d_test[:, 1], s=50, hue=y_2d_test, ax=ax[1])
ax[1].set(title="Raw Data - Test Set")
#plt.show()

## built in GP -----------------------------------------------------------------------------------------
print("--GP fixed theta -------------------------------------------------------------------------------------")

trial_fix = GaussianProcessClassifier(kernel=kernel, optimizer=None)
trial_fix.fit(X_2d_train, y_2d_train)
print("kernel n_dims, fixed theta", trial_fix.kernel.n_dims, "theta",  trial_fix.kernel.theta)
K_fix = trial_fix.kernel(X_2d_train)
print("K shape", K_fix.shape)
Z_b_fix, objects_b_fix = trial_fix.posterior_mode(K_fix, return_temporaries=True)
print("Z fix", Z_b_fix) # this is the likelihood
pi_b_fix, W_sr_b_fix, L_b_fix, b_b_fix, a_b_fix = objects_b_fix
print("pi", pi_b_fix.shape)



#posterior_mean_fix = trial_fix._posterior_mode(K_fix)
#print(trial_fix.f)

print("Kernel (initial): {}".format(trial_fix.kernel_))
print("Log Marginal Likelihood (initial) {}, {}".format(trial_fix.log_marginal_likelihood(trial_fix.kernel_.theta),
                                               trial_fix.log_marginal_likelihood_value_))
print("Theta (initial): %.3f"% trial_fix.kernel_.theta)
print("Accuracy: %.3f (initial)" % (accuracy_score(y_2d_train, trial_fix.predict(X_2d_train))))
print("Log-loss: %.3f (initial) "% (log_loss(y_2d_train, trial_fix.predict_proba(X_2d_train))))


print("--GP optim theta-------------------------------------------------------------------------------------")

trial_opt = GaussianProcessClassifier(kernel=kernel, warm_start = True)
trial_opt.fit(X_2d_train, y_2d_train)
K_opt = trial_opt.kernel(X_2d_train)

print("Kernel (optim): {}".format(trial_opt.kernel_))
print("Log Marginal Likelihood (optim) {}, {}".format(trial_opt.log_marginal_likelihood(trial_opt.kernel_.theta),
                                               trial_opt.log_marginal_likelihood_value_))
print("Theta (optim): %.3f"% trial_opt.kernel_.theta)
print("Accuracy: %.3f (optim)" % (accuracy_score(y_2d_train, trial_opt.predict(X_2d_train))))
print("Log-loss: %.3f (optim) "% (log_loss(y_2d_train, trial_opt.predict_proba(X_2d_train))))
Z_c_opt, objects_c_opt = trial_opt.posterior_mode(K_opt, return_temporaries=True)
print("Z opt", Z_c_opt) # this is the likelihood
pi_c_opt, W_sr_c_opt, L_c_opt, b_c_opt, a_c_opt = objects_c_opt
print("pi", pi_c_opt.shape)

'--------------------------------------------------------------------------------------------------------------'
print("PLOT--------------------------------------------")

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
feature_1, feature_2 = np.meshgrid(
     np.linspace(X_2d_train[:, 0].min(), X_2d_train[:, 0].max()),
     np.linspace(X_2d_train[:, 1].min(), X_2d_train[:, 1].max())
)
grid = np.vstack([feature_1.ravel(), feature_2.ravel()]).T
tree = DecisionTreeClassifier().fit(X_2d_train[:, :2], y_2d_train)
y_pred = np.reshape(tree.predict(grid), feature_1.shape)
display = DecisionBoundaryDisplay(
    xx0=feature_1, xx1=feature_2, response=y_pred
)
display.plot()

display.ax_.scatter(
    X_2d_train[:, 0], X_2d_train[:, 1], c=y_2d_train, edgecolor="black"
)
#plt.show()


print("----------------------------------------------end")
print("-- minimization grid search -----------------------------------------------------------------------------------------")

#minimize = FindMin(trial_opt)                  
#new_Xa = minimize.search_Xa(X_2d_train, y_2d_train)
#print("minim", new_Xa.shape,  new_Xa[0:10])

print("----------------------------------------------end")
print("-- update fit, reset prior params to previous posterior params----------------------------------------")

optimized_params = trial_opt.kernel_.get_params()
print("parameters kernel", optimized_params)
print("kernel gamma optimized", optimized_params["gamma"])
print("degree polyn and coef", optimized_params["pairwise_kernels_kwargs"]["degree"], optimized_params["pairwise_kernels_kwargs"]["coef0"])
print(optimized_params["metric"])
optimized_set_params = dict(itertools.islice(optimized_params.items(), 1))
print("optimized_set_params",optimized_set_params)
print(kernel.set_params(**optimized_set_params))

print("----------------------------------------------end")
print("-- optimize ----------------------------------------")

#print("bounds", trial_opt.kernel_.bounds, trial_opt.kernel_.hyperparameters, trial_opt.kernel_.bounds[:, 0])
#find = trial_opt.fit_min(X_2d_train)

Xa_bounds = np.array([(-4), (4) ])
print("Xa bounds", Xa_bounds[0])
#lb, ub = zip(*Xa_bounds)
#print("lb ub", lb, ub)
Minim = Min(X_2d_train, y_2d_train, trial_opt)
new_Xa = Minim.fit_min()
print("minim", new_Xa.shape,  new_Xa[0:10])