import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
import warnings
from torch.functional import F
from copy import copy
import itertools
from sklearn.gaussian_process.kernels import PairwiseKernel, Exponentiation, WhiteKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics.pairwise import polynomial_kernel
import pandas as pd
import torch
import gpytorch
from torch import Tensor
from torch.nn import ModuleList
from gpytorch.models import AbstractVariationalGP
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.variational import VariationalStrategy, IndependentMultitaskVariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.lazy import DiagLazyTensor
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_environment')
from wrapper import BasicWrapper
sys.path.append('C:/Users/cvcla/my_py_projects/mf_PPO')
from constants import *
from model import Actor, Critic, MLPBase
from ppo import PPO
from running_state import *
from replay_memory import *
#from utilis import *


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
output = env.multi_step(A.detach().numpy(), S, Y, patients) 
df =np.vstack([output["patients"][0][:, 2], output["Xa_post"][0]]).reshape(-1, 2) # pat is 1, Xs, Xa  
print("classes", np.sum(outcome)) 
kernel = PairwiseKernel(metric = "polynomial", pairwise_kernels_kwargs={"degree":2, "coef0":0})
print("kernel", kernel)
scaler = preprocessing.StandardScaler().fit(df)
X = scaler.transform(df)

y = outcome

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
print("TEST data - x train", X_test_torch.shape ,"y train", y_test_torch.shape, X_test_torch.detach().numpy().shape)



kernel_torch = gpytorch.kernels.PolynomialKernel(power=2)



class Model(nn.Module):
    """Custom Pytorch model for gradient optimization.
    """
    def __init__(self, kernel):
        
        super().__init__()
        # initialize weights with random numbers
        weights = torch.distributions.Uniform(-4, 4).sample((100,))
        # make weights torch parameters
        self.weights = nn.Parameter(weights)    
        self.kernel = kernel 
        self.GPc = GaussianProcessClassifier(kernel = kernel, warm_start=True)

        
            
    def fitGP(self, X, y):
        X_2d_train = X[:train_size, 0:2]; y_2d_train = y[:train_size, ]; X_2d_test = X[:train_size, 0:2]; y_2d_test = y[train_size:, ]
        X_train_torch = torch.from_numpy(X_2d_train).float(); y_train_torch = torch.from_numpy(y_2d_train).float(); X_test_torch = torch.from_numpy(X_2d_test).float(); y_test_torch = torch.from_numpy(y_2d_test).float()
        self.Xtrain_np = X_2d_train; self.Ytrain_np = y_2d_train; self.Xtrain_torch = X_train_torch; self.Ytrain_torch = y_train_torch 
        print("shapes", X_2d_train.shape, y_2d_train.shape, self.Xtrain_torch.dim(), X_train_torch.shape, y_train_torch.shape)

        self.GPc.fit(self.Xtrain_np, self.Ytrain_np)  
        print("Log Marginal Likelihood (initial) %.3f"% self.GPc.log_marginal_likelihood(self.GPc.kernel_.theta))
        print("Theta (initial): %.3f"% self.GPc.kernel_.theta)
        
        
        optimized_params = self.GPc.kernel_.get_params()
        optimized_set_gamma = dict(itertools.islice(optimized_params.items(), 1))
        print("optimized_set_params",optimized_set_gamma)
        print(kernel.set_params(**optimized_set_gamma))
        self.dic_params = self.GPc.kernel_.pairwise_kernels_kwargs
        self.dic_params.update(optimized_set_gamma) #self.GPc.kernel_.gamma
        self.dic_params.update({"metric":self.GPc.kernel_.get_params()["metric"] })
        print("pairwise_kernels_kwargs", self.dic_params)
        

    def _kernel_compute(self, X, Xstar, type):    
        # inputs are X[i]

        if type == "Kstar":
            kernel = (self.dic_params['gamma'] * torch.matmul(self.Xtrain_torch, Xstar.transpose(0, -1)) + self.dic_params['coef0'] ).pow(self.dic_params['degree'])
            print("Xstar.transpose(0, -1)", Xstar.transpose(0, -1).size)    
        elif type == "K":
            kernel = (self.dic_params['gamma'] * torch.matmul(self.Xtrain_torch, X.transpose(0, -1)) + self.dic_params['coef0']).pow(self.dic_params['degree'])
        
        return kernel
    def _fx(self, test, pi_, L_, W_sr_):
            print("shape fx torch", test.detach().numpy().shape)
            #num, var = self.GPc.post_parameters(test.detach().numpy()) 
            num, var = self._post_parameters(test.detach().numpy(), pi_, L_, W_sr_)
            print("num var shape", num.shape, var.shape)
            self.var = torch.tensor(var, requires_grad=True)
            self.num = torch.tensor(num, requires_grad=True)
            den = torch.sqrt(1+ torch.pi * self.var / 8)            
            value = torch.divide(self.num.squeeze(), den.squeeze()) #(200,)            
            return [torch.sigmoid(value), value]
    
    def _post_parameters(self, test, pi_, L_, W_sr_):      
        # Based on Algorithm 3.2 of GPML
        K_star = self.GPc.kernel(test)
        post_mean = K_star.T.dot(self.Ytrain_np - pi_)  # Line 4
        print("post_mean", post_mean.shape)
        v = solve(L_, W_sr_[:, np.newaxis] * K_star)  # Line 5
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        post_var = self.GPc.kernel.diag(test) - np.einsum("ij,ij->j", v, v)
        y_var_negative = post_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. "
                "Setting those variances to 0."
            )
            post_var[y_var_negative] = 0.0
        a_tuple = [post_mean, post_var]
        return a_tuple

    def _deriv_kern(self, Xtest):
        # compute first derivative of kernels            
        der1 = torch.zeros_like(self.Xtrain_torch)
        der2 = torch.zeros_like(self.Xtrain_torch)
        

        if self.dic_params["metric"] == "polynomial":
            for i in range(der1.shape[0]):                    
                der1[i] = self.dic_params['gamma'] *self.dic_params['degree']*self.Xtrain_torch[i]
                der2[i] = 2*self.dic_params['gamma'] *self.dic_params['degree']*self.Xtrain_torch[i]
            der11 = torch.inner(((self.dic_params['gamma']  * torch.inner(Xtest, self.Xtrain_torch)) **(self.dic_params['degree'] - 1)), der1.T)
            der22 = torch.inner(((self.dic_params['gamma'] *torch.inner(Xtest, self.Xtrain_torch)) **(2*self.dic_params['degree'] - 1)), der2.T)
        der_tuple = [der11, der22] # shape of each is n,self.opt_params
        return der_tuple
    
    def _objective(self, Xtest):
        # computes objective function   
        test = Xtest.reshape(-1, 1)
        test_ = torch.column_stack((self.Xtrain_torch[:, 0], test))
        
        K_opt = self._kernel_compute(self.Xtrain_torch, test, type= "K")
        Z_c_opt, objects_c_opt = self.GPc.posterior_mode(K_opt.detach().numpy(), return_temporaries=True)
        pi_c_opt, W_sr_c_opt, L_c_opt, b_c_opt, a_c_opt = objects_c_opt
        

        values_for_min = torch.zeros((self.Xtrain_torch.shape[0],self.Xtrain_torch.shape[1]))
        for i in range(values_for_min.shape[0]):
            sigma_x = self._fx(test_, pi_c_opt, L_c_opt, W_sr_c_opt)[0][i] # now scalar ()

            fx = self._fx(test_, pi_c_opt, L_c_opt, W_sr_c_opt)[1][i] 
            der2_loglik = (pi_c_opt * (1 - pi_c_opt))[i]
            W_inv = - (1/der2_loglik)

            sum_W_K = W_inv + self._kernel_compute(self.Xtrain_torch, test_[i], type= "Kstar")
            print("kstar", self._kernel_compute(self.Xtrain_torch, test_[i], type= "Kstar").shape)
            b = 1/sum_W_K
            print("test_[i]", test_[i].shape)
            a =  (self.Ytrain_torch[i] - pi_c_opt[i]).reshape(-1, 1) 
            t1 = -(torch.pi /8) *self._kernel_compute(self.Xtrain_torch, test_[i], type= "Kstar")*b
            t2 = (1/(1+(torch.pi/8)*self.var[i])).reshape(-1, 1)     
            t3 = torch.sqrt(self.var[i]).reshape(-1, 1)
            t = t1* t2*t3
            print("shape der", self._deriv_kern(test_)[1][i].shape, t.shape, t1.shape, t2.shape, t3.shape)
            der_fx = (a * (torch.sqrt(t2))).T*(t*self._deriv_kern(test_)[1][i] + self._deriv_kern(test_)[1][i])
            x = sigma_x * (der_fx.squeeze() + (1-sigma_x)*fx)
            values_for_min[i]=x
        return values_for_min

         
    

    def forward(self):
        """Implement function to be optimised. In this case, an exponential decay
        function (a + exp(-k * X) + b),
        """        
        Xtest = self.weights
    
        return self._objective(Xtest)

def training_loop(model, optimizer, n=1000):
    "Training loop for torch model."
    losses = []
    for i in range(n):
        y = torch.ones([100,1])
        preds = model.forward()
        loss = F.mse_loss(preds, y).sqrt()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)  
    return losses


# instantiate model
m = Model(kernel)
m.fitGP(X, y)

# Instantiate optimizer
opt = torch.optim.Adam(m.parameters(), lr=0.001)
losses = training_loop(m, opt)
plt.figure(figsize=(14, 7))
plt.plot(losses)
print(m.weights.shape)



'''
GPc = GaussianProcessClassifier(kernel = kernel)
GPc.fit(Xa_prime.reshape(-1,1), outcome.reshape(-1,1)) 
weights = torch.distributions.Uniform(0, 0.1).sample((200,1))
print(weights.shape)
'''
'''
from skopt import gp_minimize

res = gp_minimize(minimize.FOC_i_tensor(self.data, Xtest),                  # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed

'''
print("----------------------------------------------end")
print("-- gpytorch -----------------------------------------------------------------------------------------")


'''

use_priors = True
class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x):
        variational_distribution = variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(train_x.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self,
                                train_x,
                                variational_distribution,
                                learn_inducing_locations=False)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.PolynomialKernel(power=2)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

def training_loop(X, model, likelihood, n_iter=1000):
    "Training loop for torch model."
    # Put the model into training mode
    

    model.train()
    likelihood.train()
    # Use the Adam optimizer, with learning rate set to 0.1
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    y = y_train_torch
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=y.size(0), lr=0.1)
    # Use the negative marginal log-likelihood as the loss function
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())
    losses = []
    param_kernel = []
    for i in range(n_iter):
        # Set the gradients from previous iteration to zero
        #optimizer.zero_grad()
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()
        
        preds = model(X)
        loss = -mll(preds, y)
        loss.backward()
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()

        for name, param in model.state_dict().items():
            if name == "covar_module.raw_offset":
                param_kernel.append(param)
            elif name == "model.covar_module.raw_outputscale":
                param_kernel.append(param)
        if i % 200 == 0:
            
            
    
            print('Iter {}/{} - Loss: {} lengthscale: noise: '.format(i, n_iter, loss.item()))
            print("model.covar_module.raw_outputscale", param_kernel[i])
        
        losses.append(loss)  
    
    
        
    
    return losses




def testing_loop(testX, trainX, trainY, model, likelihood):

    # Put the model into evaluation mode
    model.eval()
    likelihood.eval()

    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    # See https://arxiv.org/abs/1803.06058
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Obtain the predictive mean and covariance matrix
        f_preds = model(testX)
        f_mean = f_preds.mean
        f_cov = f_preds.covariance_matrix
        
        # Make predictions by feeding model through likelihood
        observed_pred = likelihood(model(testX))
        
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(trainX.numpy(), trainY.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(testX.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(testX.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])


# Initialize model and likelihood
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
model = GPClassificationModel(X_train_torch)

losses = training_loop(X_train_torch, model, likelihood)
   



'''