from sklearn.gaussian_process.kernels import PairwiseKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.metrics.pairwise import polynomial_kernel
from kernels import DotProduct, Exponentiation, RBF_1
import pandas as pd
from scipy.special import erf, expit as sigmoid
import os
import numpy as np
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_game')
from wrapper import BasicWrapper
import json, urllib


sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/PPO_2')
from model import Actor, Critic
from ppo import PPO
from constants import get_args
from derivate import Derivate

from make_plots import fig1

'''
directory to save plots
'''
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'GP_plots/')
if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

'''
initialize environment, bits of PPO
'''
args = get_args()
env = BasicWrapper()
n_input = env.observation_size
n_output = env.action_size

actor = Actor(n_input, n_output, args.n_hidden)
critic = Critic(n_input, args.n_hidden)
ppo_agent = PPO(env, args, actor, critic)

patients, S = env.reset() # S tensor
A = env.sample_random_action()
S_prime, R, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, is_done = env.step(A, S.detach().numpy())

'''
arrange objcts in a df
'''
patients = patients[:,1:3]
start_state = S
Xa_initial = patients[:, 1]
Xs_initial = patients[:, 0]
levels = ["L", "M", "H"]
rho_init = np.random.beta(7, 3, size=n_input)
Xa_post = env.intervention(Xa_initial, rho_init, which=2)
df = pd.DataFrame(data={'Xa_reset':  Xa_initial, 'Xa_post':  Xa_post, 'states': start_state, "Outcome": outcome})
df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                            bins=[0, 0.4, 0.8, 1],
                                            labels=levels)))



'''
compute min objective, derivative and posterior parameters
'''
trial = Derivate(trainX = df['Xa_reset'].to_numpy().reshape(-1, 1), 
                 trainY = df['Outcome'].to_numpy().reshape(-1, 1), 
                 testX = df['Xa_post'].to_numpy().reshape(-1, 1), specify_kernel='linear')

parameters = trial.params()
mean_star = parameters[0].squeeze(1)
print("mean_star", mean_star.shape)
var_star = parameters[1]
print("var_star", var_star.shape)
kappa = 1.0 / np.sqrt(1.0 + np.pi * var_star / 8)
predicted_proba = sigmoid(kappa * mean_star)
print("predicted_proba", predicted_proba.shape)
df["Probability"] = predicted_proba
df["out_1"] = np.where(predicted_proba<0.5, 0, 1)
df.sort_values(by=['Xa_reset'])


'''
make some plots
'''

fig1(df, "predicted_values", results_dir)