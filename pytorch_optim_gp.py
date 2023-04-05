import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import warnings
from torch.functional import F
from copy import copy
from sklearn.gaussian_process.kernels import PairwiseKernel, Exponentiation, WhiteKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.metrics.pairwise import polynomial_kernel
import pandas as pd
import torch
from torch import Tensor
from torch.nn import ModuleList
from GP_param_train import GaussianProcessClassifierLaplace
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_environment')
from wrapper import BasicWrapper
sys.path.append('C:/Users/cvcla/my_py_projects/mf_PPO')
from constants import *
from model import Actor, Critic, MLPBase
from ppo import PPO
from running_state import *
from replay_memory import *


## suppose in the exploration part, the intervention funciton is deterministic


# make df to store X_a values
df = pd.DataFrame()


args = get_args()
env = BasicWrapper()
actor = Actor(env.observation_size, env.action_size, args.n_hidden)
critic = Critic(env.observation_size, args.n_hidden)  
MLPBase_model = MLPBase(env.observation_size, env.action_size, env.action_size) #what 3rd arg?
#GP_transition = StepGP(args, kernel_choice = linear) 

replay_buffer = ReplayBuffer(capacity=args.buffer_capacity,
                                observation_shape= env.observation_size,
                                action_dim=env.action_size)
running_state = ZFilter((env.observation_size,), clip=5)

ppo_agent = PPO(env, args, actor, critic, MLPBase_model) 

patients, S = env.reset() # S tensor
A = env.sample_random_action()
S_prime, R, pat, s_LogReg, r_LogReg, Xa, Xa_prime, outcome, done = env.step(A, S.detach().numpy())   

kernel = RBF() + WhiteKernel(noise_level=0.5)


class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, kernel):
        
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(0, 0.1).sample((200,))
        # make weights torch parameters
        self.weights = nn.Parameter(weights)    
        self.kernel = kernel 
        self.GPc = GaussianProcessClassifierLaplace(kernel = kernel)
            
    def fitGP(self, Xa, Y):
        self.Xa = Xa
        self.Y = Y
        self.GPc.fit(self.Xa.reshape(-1,1), self.Y.reshape(-1,1))  
        self.lengthscale = self.GPc.kernel_.theta[0]

         
    def post_parameters(self):
        """Computes parameters of the posterior distributions
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Query points where the GP is evaluated for classification.
        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X, values are from ``classes_``
        """


        # As discussed on Section 3.4.2 of GPML, for making hard binary
        # decisions, it is enough to compute the MAP of the posterior and
        # pass it through the link function

        input = torch.from_numpy(self.Xa)
        weights = torch.distributions.Uniform(0, 0.1).sample((200,))
        K_star = RBFCovariance(input, weights, 0.5,
            lambda input, weights: covar_dist(input, weights, square_dist=True, diag=False),
        )  
        return K_star
        
        '''
        #K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
        post_mean = K_star.T.dot(self.y_train_ - self.pi)  # Algorithm 3.2,Line 4

        v = solve(self.L, self.W_sr * K_star)  # Line 5 #W_sr_[:, np.newaxis]
        v1 = v.T.dot(v)
        v2 = np.diag(v1)
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        #post_var = np.ones(self.X_train_.shape[0])
        post_var = self.kernel_.diag(X)
        post_var -= np.einsum("ij,ji->i", v.T, v)

        y_var_negative = post_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. "
                "Setting those variances to 0."
            )
            post_var[y_var_negative] = 0.0
        
        a_tuple = [post_mean, post_var]

        return a_tuple
        '''

    def forward(self):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """
        Xtest = self.weights
        return self.GPc.f_x(Xtest)[0]
    
def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        y = torch.ones([200,1])
        preds = model.forward()
        loss = F.mse_loss(preds, y).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)  
    return losses

def sq_dist(x1, x2, x1_eq_x2=False):
    # TODO: use torch squared cdist once implemented: https://github.com/pytorch/pytorch/pull/25799
    adjustment = x1.mean(-1, keepdim=True) #-2
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        x2, x2_norm, x2_pad = x1, x1_norm, x1_pad
    else:
        x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-1, 0)) #(-2, -1)

    if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
        res.diagonal(dim1=-1, dim2=0).fill_(0) # (dim1=-2, dim2=-1)

    # Zero out negative values
    return res.clamp_min_(0)


def RBFCovariance( x1, x2, lengthscale, sq_dist_func):
    x1_ = x1.div(lengthscale)
    x2_ = x2.div(lengthscale)
    unitless_sq_dist = sq_dist_func(x1_, x2_)
    covar_mat = unitless_sq_dist.div_(-2.0).exp_()
    
    return covar_mat


def dist(x1, x2, x1_eq_x2=False):
    # TODO: use torch cdist once implementation is improved: https://github.com/pytorch/pytorch/pull/25799
    res = sq_dist(x1, x2, x1_eq_x2=x1_eq_x2)
    return res.clamp_min_(1e-30).sqrt_()


def covar_dist(
        
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> Tensor:
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.
        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)
        :param square_dist:
            If True, returns the squared distance rather than the standard distance. (Default: False.)
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:
            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)
        res = None

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                return torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
            else:
                res = torch.linalg.norm(x1 - x2, dim=-1)  # 2-norm by default
                return res.pow(2) if square_dist else res
        else:
            dist_func = sq_dist if square_dist else dist
            return dist_func(x1, x2, x1_eq_x2)
        
      
# instantiate model
m = Model(kernel)
m.fitGP(Xa_prime, outcome)
print("lengthscale", m.lengthscale)
print("K", m.post_parameters())
# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.001)

'''
losses = training_loop(m, opt)
plt.figure(figsize=(14, 7))
plt.plot(losses)
print(m.weights)
'''


'''
GPc = GaussianProcessClassifierLaplace(kernel = kernel)
GPc.fit(Xa_prime.reshape(-1,1), outcome.reshape(-1,1)) 
weights = torch.distributions.Uniform(0, 0.1).sample((200,1))
print(weights.shape)
'''