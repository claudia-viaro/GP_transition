
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import time
from datetime import datetime
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
from utilis import *
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

episode = 1
start = datetime.now().strftime("%d/%m/%Y %H:%M:%S"); start_time = time.time()
patients, S, Y = env.reset() # S tensor, patients (n, 4)
Xa_pre = patients[:, 2]
total_reward = 0; total_rewardLR = 0; count_iter = 0
Done = False

while not Done: #not done:
    #max_done += 1
    count_iter +=1 # count transitions in a trajectory        
    
    A = env.sample_random_action()
    output = env.step_wrapper(A, S, Y, Xa_pre)   
    S = running_state(output["rho"][0]); pat_df = output["patients"][0]; Y = output["outcome"][0]; Xa_pre = pat_df[:, 2]; Done = output["done"][0]; reward = output["reward"][0]     
    print("check shapes S", S.shape, S[0:5], "patients", pat_df.shape, pat_df[:, 2][0:5], "Y", Y.shape, "Xa pre", Xa_pre.shape,  "Done", Done, "rho", output["rho"][0][0:5])