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
from GP_param_train import GaussianProcessClassifierLaplace
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

patients, S = env.reset() # S tensor
A = env.sample_random_action()
S_prime, R, pat, s_LogReg, r_LogReg, Xa_prime, outcome, done = env.step(A, S.detach().numpy(), patients) 
print("classes", np.sum(outcome)) 
kernel = PairwiseKernel(metric = "polynomial", pairwise_kernels_kwargs={"degree":2, "coef0":0})
print("Ply kernel", kernel)
#kernel = RBF() + WhiteKernel(noise_level=0.5)
df =np.vstack([pat[:, 1], Xa_prime]).reshape(-1, 2) # pat is 1, Xs, Xa
#df = pd.DataFrame(data={'Xs':pat[:, 1], 'Xa': Xa_prime}).to_numpy()
XS = np.random.normal(0,1, size=(200,1))
print("prescal" , XS.shape,Xa_prime.shape )
pre_scaling = np.column_stack([XS, Xa_prime]).reshape(-1, XS.shape[1]+1) # this df potentially can grow in dim 1
print("prescaling", pre_scaling)
# if no scale
#X = df
# if scale data
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
print("TEST data - x train", X_test_torch.shape ,"y train", y_test_torch.shape)


fig, ax = plt.subplots(
    nrows=1, ncols=2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained"
)
sns.scatterplot(x=X_2d_train[:, 0], y=X_2d_train[:, 1], s=50, hue=y_2d_train, ax=ax[0])
ax[0].set(title="Raw Data - Training Set")
sns.scatterplot(x=X_2d_test[:, 0], y=X_2d_test[:, 1], s=50, hue=y_2d_test, ax=ax[1])
ax[1].set(title="Raw Data - Test Set")
plt.show()

## built in GP -----------------------------------------------------------------------------------------
print("--Built in GP-----------------------------------------------------------------------------------------")

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



trial_opt = GaussianProcessClassifier(kernel=kernel)
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
plt.show()


print("----------------------------------------------end")
print("-- minimization part -----------------------------------------------------------------------------------------")
print("self.Xtrain[: , :-1]", X_2d_train[: , :-1].shape)
print(np.random.normal(0,1, size=(200,1)).shape)


minimize = FindMin(trial_opt)
                   
new_Xa = minimize.search_Xa(X_2d_train, y_2d_train)

print("minim", new_Xa.shape,  new_Xa[0:10])
'''
trial = np.random.uniform(-4, 4,100)
print("shapes trial Xa* Xs", trial.shape, X_2d_train[:, 1].shape)
trial_df = np.vstack([trial, X_2d_train[:, 1]]).reshape(-1, 2)
print("trial_df - Xa* Xs", trial_df.shape)
predicted = trial_opt.predict_proba(X_2d_test) # not needed I think
#print("predicted shape", predicted.shape)

K_trial = trial_opt.kernel(trial_df)
print("kernel trial", K_trial.shape)
mean, var = trial_opt.post_parameters(trial_df)
print("mean, var", mean.shape, var.shape)
print("sample mean and var", mean[0:5], var[0:5])
den = np.sqrt(1+ np.pi * var / 8)
        
value = np.divide(mean.squeeze(), den.squeeze()) #(200,)
f_x = sigmoid(value)
print("f_x", f_x.shape, f_x[0:5])
'''

optimized_params = trial_opt.kernel_.get_params()
print("parameters kernel", optimized_params)
print("kernel gamma optimized", optimized_params["gamma"])
print("degree polyn and coef", optimized_params["pairwise_kernels_kwargs"]["degree"], optimized_params["pairwise_kernels_kwargs"]["coef0"])
print(optimized_params["metric"])
optimized_set_params = dict(itertools.islice(optimized_params.items(), 1))
print("optimized_set_params",optimized_set_params)
print(kernel.set_params(**optimized_set_params))


der1 = np.zeros((X_2d_train.shape[0],X_2d_train.shape[1]))

for i in range(der1.shape[0]):
            
            der1[i] = optimized_params["gamma"]*optimized_params["pairwise_kernels_kwargs"]["degree"]*X_2d_train[i]




'''
der1 = np.zeros((100,2))
der2 = np.zeros((100,2))
print("shape init der1", der1.shape[0])
print("other objs", X_2d_train[2])
print("other", np.inner(trial_df, X_2d_train).shape)

for i in range(der1.shape[0]):
      der1[i] = optimized_params["gamma"]*optimized_params["pairwise_kernels_kwargs"]["degree"]*X_2d_train[i]
      der2[i] = 2*optimized_params["gamma"]*optimized_params["pairwise_kernels_kwargs"]["degree"]*X_2d_train[i]
print("der1", der1.shape)
#der_tuple = [np.array(der1*np.inner(trial_df, X_2d_train) **(optimized_params["pairwise_kernels_kwargs"]["degree"] - 1)).squeeze(), np.array(der2*np.inner(trial_df, X_2d_train) **(2*optimized_params["pairwise_kernels_kwargs"]["degree"] - 1) ).squeeze()]  
der22 = np.inner(((optimized_params["gamma"] * np.inner(trial_df, X_2d_train)) **(optimized_params["pairwise_kernels_kwargs"]["degree"] - 1)), der1.T)
der11 = np.inner(((optimized_params["gamma"]*np.inner(trial_df, X_2d_train)) **(2*optimized_params["pairwise_kernels_kwargs"]["degree"] - 1)), der2.T)
print("derivative shapes", der11.shape, der22.shape)


values_for_min = np.zeros((100,2))
for i in range(values_for_min.shape[0]):
      print("one instane", trial_df[i].shape)
      sigma_x = f_x[i] # now scalar ()
      print("sigma shape", sigma_x.shape)
      fx = value[i] 
      print("fx shape", f_x.shape)
      der2_loglik = (pi_c_opt * (1 - pi_c_opt))[i]
      print("der2_loglik", der2_loglik.shape)
      W_inv = - (1/der2_loglik)
      sum_W_K = W_inv + trial_opt.kernel(trial_df[i])
      print("sum_W_K", sum_W_K.shape)
      b = 1/sum_W_K
      print("b", b.shape) 
      a =  (y_2d_train[i] - pi_c_opt[i]).reshape(-1, 1) 
      print("a", a.shape) 
      t1 = -(np.pi /8) *trial_opt.kernel(trial_df[i])*b
      print("t1", t1.shape) 
      t2 = (1/(1+(np.pi/8)*var[i])).reshape(-1, 1)     
      print("t2", t2.shape)        
      t3 = np.sqrt(var[i]).reshape(-1, 1)
      print("t3", t3.shape) 
      t = t1* t2*t3
      der_fx = (a * (np.sqrt(t2)))*(t*der22[i] + der11[i])
      print("der_fx", der_fx.shape)
      x = sigma_x * (der_fx.squeeze() + (1-sigma_x)*fx)
      print("x", x.shape)
      values_for_min[i]=x
print("values_for_min", values_for_min.shape, type(values_for_min), values_for_min[0:5], values_for_min[:, 0])
val = np.zeros_like(y_2d_train.reshape(-1,1))
        
i = 0
for i in range(5):
      trial = np.random.uniform(-4, 4,200)
      #values = np.column_stack((values, trial))
      b = self.FOC_i(trial.reshape(-1, 1)).squeeze()

      val = np.column_stack((val,b))
      i += 1
      val = np.delete(val, 0,1)

outcome =  np.array(values).min(axis=1)
def f_x (self, trial_df):
        
        num, var, _ = self.trial_opt.post_parameters(trial_df)
    
        den = np.sqrt(1+ np.pi * var / 8)
        
        value = np.divide(num.squeeze(), den.squeeze()) #(200,)
        
        return [sigmoid(value), value]


def derivate_kern(self, trial_df):
        
        
      der1 = np.zeros((100,2))
      der2 = np.zeros((100,2))

      for i in range(der1.shape[0]):
            der1[i] = optimized_params["gamma"]*optimized_params["pairwise_kernels_kwargs"]["degree"]*X_2d_train[i]
            der2[i] = 2*optimized_params["gamma"]*optimized_params["pairwise_kernels_kwargs"]["degree"]*X_2d_train[i]
      der22 = np.inner(((optimized_params["gamma"] * np.inner(trial_df, X_2d_train)) **(optimized_params["pairwise_kernels_kwargs"]["degree"] - 1)), der1.T)
      der11 = np.inner(((optimized_params["gamma"]*np.inner(trial_df, X_2d_train)) **(2*optimized_params["pairwise_kernels_kwargs"]["degree"] - 1)), der2.T)
      der_tuple = [der11, der22] # shape of each is n,2

       

      return der_tuple
def FOC_i(self, Xtest):
        # computes derivative function
        
        
      values_for_min = np.zeros((100,2))
      for i in range(values_for_min.shape[0]):
            sigma_x = f_x[i] # now scalar ()
            fx = value[i] 
            der2_loglik = (pi_c_opt * (1 - pi_c_opt))[i]
            W_inv = - (1/der2_loglik)
            sum_W_K = W_inv + trial_opt.kernel(trial_df[i])
            b = 1/sum_W_K
            a =  (y_2d_train[i] - pi_c_opt[i]).reshape(-1, 1) 
            t1 = -(np.pi /8) *trial_opt.kernel(trial_df[i])*b
            t2 = (1/(1+(np.pi/8)*var[i])).reshape(-1, 1)     
            t3 = np.sqrt(var[i]).reshape(-1, 1)
            t = t1* t2*t3
            der_fx = (a * (np.sqrt(t2)))*(t*der22[i] + der11[i])
            x = sigma_x * (der_fx.squeeze() + (1-sigma_x)*fx)
            print("x", x.shape)
            values_for_min[i]=x
      return values_for_min
def find_min(self):
         
        values = np.zeros_like(self.y_train_.reshape(-1,1))
        
        i = 0
        for i in range(5):
            trial = np.random.uniform(-4, 4,200)
            #values = np.column_stack((values, trial))
            b = self.FOC_i(trial.reshape(-1, 1)).squeeze()

            values = np.column_stack((values,b))
            i += 1
        values = np.delete(values, 0,1)
    
        return np.array(values).min(axis=1)
'''


## custom GP -----------------------------------------------------------------------------------------
print("--Custom GP -----------------------------------------------------------------------------------------")

''' check posterior mode function
GPc_fix = GaussianProcessClassifierLaplace(kernel = kernel, optimizer=None)
GPc_fix.fit(X_2d_train, y_2d_train)
#ff = np.zeros_like(y_2d_train, dtype=np.float64).reshape(-1, 1)
print("kernel n_dims", GPc_fix.kernel.n_dims, "theta",  GPc_fix.kernel.theta)
K_c_fix = GPc_fix.kernel(X_2d_train)
print("K shape", K_c_fix.shape)
Z_c_fix, objects_c_fix = GPc_fix._posterior_mode(K_c_fix)
print("Z", Z_c_fix) # this is the likelihood
pi_c_fix, W_sr_c_fix, L_c_fix, b_c_fix, a_c_fix = objects_c_fix
print("pi", pi_c_fix.shape)

GPc_opt = GaussianProcessClassifierLaplace(kernel = kernel)


#GPc_opt.fit(X_2d_train, y_2d_train)

print("kernel n_dims", GPc_opt.kernel.n_dims, "theta optim",  GPc_opt.kernel.theta)
K_c_opt = GPc_opt.kernel(X_2d_train)
print("K shape optim", K_c_opt.shape)
Z_c_opt, objects_c_opt = GPc_fix._posterior_mode(K_c_opt)
print("Z optim", Z_c_opt) # this is the likelihood
pi_c_opt, W_sr_c_opt, L_c_opt, b_c_opt, a_c_opt = objects_c_opt
print("pi", pi_c_opt.shape)

print("Log Marginal Likelihood (initial): %.3f"
      % GPc_fix.log_marginal_likelihood(GPc_fix.kernel_.theta))
print("Log Marginal Likelihood (optimized): %.3f"
      % GPc_opt.log_marginal_likelihood(GPc_opt.kernel_.theta))
print("Theta (initial)", GPc_fix.kernel_.theta)
#print("Accuracy: %.3f (initial) %.3f (optimized)" % (accuracy_score(outcome, GPc_fix.post_parameters(df)),accuracy_score(outcome, GPc_opt.post_parameters(df))))

p_mean_c_fix, p_var_c_fix, predictions_c_fix = GPc_fix.post_parameters(X_2d_train)
print("Posterior parameters",p_mean_c_fix.shape, p_var_c_fix.shape, predictions_c_fix.shape, np.sum(predictions_c_fix))
p_mean_c_opt, p_var_c_opt, predictions_c_opt = GPc_opt.post_parameters(X_2d_train)
print("Posterior parameters",p_mean_c_opt.shape, p_var_c_opt.shape, predictions_c_opt.shape, np.sum(predictions_c_opt))

## check error
f_optim = p_mean_c_opt
a_optim = a_c_opt
L_optim = L_c_opt

lml_c_opt = (
                -0.5 * a_optim.T.dot(f_optim)
                - np.log1p(np.exp(-(y_2d_train.reshape(-1,1) * 2 - 1) * f_optim)).sum()
                - np.log(np.diag(L_optim)).sum())
#trial = GPc.fit_Xa()
print("length scale GPc optim", GPc_opt.kernel_.get_params())
print("length scale GPc fix", GPc_fix.kernel_.get_params())

#Xa_post = GPc_opt.find_min() 



GPc_fix = GaussianProcessClassifierLaplace(kernel = kernel, optimizer=None)
K_c_fix = GPc_fix.kernel(X_2d_train)


f = np.zeros_like(y_2d_train, dtype=np.float64).reshape(-1, 1)
# Use Newton's iteration method to find mode of Laplace approximation
log_marginal_likelihood = -np.inf
#for _ in range(2):
for i in range(10):
      print("i", i)
      # Line 4
      pi = expit(f)
      W = pi * (1 - pi)
      # Line 5
      W_sr = np.sqrt(W)
      W_sr_K = W_sr * K_c_fix

      #W_sr_K = W_sr[:, np.newaxis] * K
      B = np.eye(W.shape[0]) + W_sr_K * W_sr
      tweakB = nearestPD(B)
      L = cholesky(tweakB, lower=True)

      # Line 6
      b = W * f + (y_2d_train - pi)
      # Line 7
      cho = cho_solve((L, True), W_sr_K.dot(b))
      cho_prod = W_sr * cho
      a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
      # Line 8
      f = K_c_fix.dot(a)
      print("f", f)
      time.sleep(5)

      # Line 10: Compute log marginal likelihood in loop and use as
      #          convergence criterion
      #print("np.log(np.diag(L)).sum()", np.log(np.diag(L)).sum().sum(), np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum(), a.T.dot(f))
      lml = (
            -0.5 * a.T.dot(f)
            - np.log1p(np.exp(-(y_2d_train * 2 - 1) * f)).sum()
            - np.log(np.diag(L)).sum()
      )
      print("lml shape", lml.shape)
      print("lml", lml)
      print("lml", lml - log_marginal_likelihood)
      # Check if we have converged (log marginal likelihood does
      # not decrease)
      # XXX: more complex convergence criterion
      if lml - log_marginal_likelihood < 1e-10:
            print("lml 1", i, log_marginal_likelihood)
            break
      log_marginal_likelihood = lml



print("Custom 2D GP with random data")

X_2d = rng.uniform(0, 5, (100, 2))
y_2d = np.array(X_2d[:, 0] > 2.5, dtype=int)
X_2d_train = X_2d[:train_size, ]; y_2d_train = y_2d[:train_size, ]; X_2d_test = X_2d[train_size:, ]; y_2d_test = y_2d[train_size:, ]
print("Shapes", "x train", X_2d.shape ,X_2d_train.shape, "y train", y_2d.shape, y_2d_train.shape)
GPc_fix_1 = GaussianProcessClassifierLaplace(kernel = kernel, optimizer=None)
print(".n_dims", GPc_fix_1.kernel.n_dims, "theta",  GPc_fix_1.kernel.theta)
K = GPc_fix_1.kernel(X_2d_train)
print("K shape", K.shape)

GPc_fix_1.fit(X_2d_train, y_2d_train.reshape(-1,1))
#print("theta", GPc.kernel_.theta[0])

GPc_opt_2 = GaussianProcessClassifierLaplace(kernel = kernel)
GPc_opt_2.fit(X_2d_train, y_2d_train.reshape(-1,1))


p_mean_fix_1, p_var_fix_1, predictions_fix_1 = GPc_fix_1.post_parameters(X_2d_train)
print(p_mean_fix_1.shape, p_var_fix_1.shape, predictions_fix_1.shape, np.sum(predictions_fix_1))
p_mean_opt_1, p_var_opt_1, predictions_opt_1 = GPc_opt_2.post_parameters(X_2d_train)
print(p_mean_opt_1.shape, p_var_opt_1.shape, predictions_opt_1.shape, np.sum(predictions_opt_1))

## check error
f_optim_1 = GPc_opt_2.f_cached 
a_optim_1 = GPc_opt_2.a
L_optim_1 = GPc_opt_2.L 
lml_optim_1 = (
                -0.5 * a_optim_1.T.dot(f_optim_1)
                - np.log1p(np.exp(-(y_2d_train* 2 - 1) * f_optim_1)).sum()
                - np.log(np.diag(L_optim_1)).sum())




print("Log Marginal Likelihood (initial): %.3f"
      % GPc_fix_1.log_marginal_likelihood(GPc_fix_1.kernel_.theta))
print("Log Marginal Likelihood (optimized): %.3f"
      % GPc_opt_2.log_marginal_likelihood(GPc_opt_2.kernel_.theta))
print("Theta (initial): %.3f"% GPc_fix_1.kernel_.theta)
print("Theta (optimized): %.3f %.3f"% (GPc_opt_2.kernel_.theta, GPc_opt_2.kernel.theta[0]))
print("Accuracy: %.3f (initial) %.3f (optimized)"
      % (accuracy_score(y_2d, GPc_fix_1.post_parameters(X_2d)),
         accuracy_score(y_2d, GPc_opt_2.post_parameters(X_2d))))
print("Log Marginal Likelihood (initial): %.3f"
      % GPc_fix_1.log_marginal_likelihood_value_)
print("Log Marginal Likelihood (optimized): %.3f"
      % GPc_opt_2.log_marginal_likelihood_value_)

print("----------------------------------------------end")
'''