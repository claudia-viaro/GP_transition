import math
import pandas as pd
import torch
import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
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

Xa_t = torch.from_numpy(Xa)
Xa_prime_t = torch.from_numpy(Xa_prime)

class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


# Initialize model and likelihood
model = GPClassificationModel(Xa_t)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()