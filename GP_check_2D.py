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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

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
kernel = PairwiseKernel(metric = "polynomial")
#kernel = RBF() + WhiteKernel(noise_level=0.5)
df = pd.DataFrame(data={'Xa': Xa_prime, 'Xs':pat[:, 1]})

## custom GP -----------------------------------------------------------------------------------------
print("Custom GP")
GPc_fix = GaussianProcessClassifierLaplace(kernel = kernel, optimizer=None)
GPc_fix.fit(df, outcome.reshape(-1,1))
#print("theta", GPc.kernel_.theta[0])

GPc_opt = GaussianProcessClassifierLaplace(kernel = kernel)
GPc_opt.fit(df, outcome.reshape(-1,1))

print("Log Marginal Likelihood (initial): %.3f"
      % GPc_fix.log_marginal_likelihood(GPc_fix.kernel_.theta))
print("Log Marginal Likelihood (optimized): %.3f"
      % GPc_opt.log_marginal_likelihood(GPc_opt.kernel_.theta))
print("Theta (initial): %.3f"% GPc_fix.kernel_.theta)
print("Theta (optimized): %.3f %.3f"% (GPc_opt.kernel_.theta), GPc_opt.kernel.theta[0])
print("Accuracy: %.3f (initial) %.3f (optimized)"
      % (accuracy_score(outcome, GPc_fix.post_parameters(df)),
         accuracy_score(outcome, GPc_opt.post_parameters(df))))

#trial = GPc.fit_Xa()
print("----------------------------------------------end")

## built in GP -----------------------------------------------------------------------------------------
print("Built in GP")
model_fix = GaussianProcessClassifier(kernel=kernel, optimizer=None)
model_fix.fit(df, outcome)

model_opt = GaussianProcessClassifier(kernel=kernel)
model_opt.fit(df, outcome)

print("Log Marginal Likelihood (initial): %.3f"
      % model_fix.log_marginal_likelihood(model_fix.kernel_.theta))
print("Log Marginal Likelihood (optimized): %.3f"
      % model_opt.log_marginal_likelihood(model_opt.kernel_.theta))
print("Theta (initial): %.3f"% model_fix.kernel_.theta)
print("Theta (optimized): %.3f"% model_opt.kernel_.theta)
print("Accuracy: %.3f (initial) %.3f (optimized)"
      % (accuracy_score(outcome, model_fix.predict(df)),
         accuracy_score(outcome, model_opt.predict(df))))

#print("Log-loss: %.3f (initial) %.3f (optimized)"
#      % (log_loss(y[:train_size], model_fix.predict_proba(X[:train_size])[:, 1]),
#         log_loss(y[:train_size], model_opt.predict_proba(X[:train_size])[:, 1])))
print("----------------------------------------------end")

print("Check trial GP")

train_size = 50
rng = np.random.RandomState(0)
X = rng.uniform(0, 5, 100)[:, np.newaxis]
y = np.array(X[:, 0] > 2.5, dtype=int)
print("x train", X.shape ,"y train", y.shape)
#print(X[:train_size][:, 1])


trial_fix = GaussianProcessClassifier(kernel=kernel, optimizer=None)
trial_fix.fit(df, outcome)

trial_opt = GaussianProcessClassifier(kernel=kernel)
trial_opt.fit(df, outcome)

print("Log Marginal Likelihood (initial): %.3f"
      % trial_fix.log_marginal_likelihood(trial_fix.kernel_.theta))
print("Log Marginal Likelihood (optimized): %.3f"
      % trial_opt.log_marginal_likelihood(trial_opt.kernel_.theta))
print("Theta (initial): %.3f"% trial_fix.kernel_.theta)
print("Theta (optimized): %.3f"% trial_opt.kernel_.theta)
print("Accuracy: %.3f (initial) %.3f (optimized)"
      % (accuracy_score(outcome, trial_fix.predict(df)),
         accuracy_score(outcome, trial_opt.predict(df))))

print("Log-loss: %.3f (initial) %.3f (optimized)"
      % (log_loss(y[:train_size], trial_fix.predict_proba(X[:train_size])[:, 1]),
         log_loss(y[:train_size], trial_opt.predict_proba(X[:train_size])[:, 1])))

print("----------------------------------------------end")
