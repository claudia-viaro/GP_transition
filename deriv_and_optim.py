import torch
import numpy as np
import os
import sklearn
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, expit as sigmoid
import pandas as pd
from sklearn.gaussian_process.kernels import PairwiseKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.metrics.pairwise import polynomial_kernel
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import (RationalQuadratic,Exponentiation)
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessClassifier

from kernels import DotProduct, Exponentiation, RBF_1
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/toy_game')
from wrapper import BasicWrapper



class Derivate(object):
    """

    """
    def __init__(self, trainX, trainY, exponent = 2):
        ## dataser
        self.trainX = trainX
        self.trainY = trainY


        ## kernel
        self.exponent = exponent
        self.theta = 1
        
        

        self.index = 0
        self.is_filled = False

    def init_kernel(self):
        lin_k= DotProduct() + WhiteKernel()
        kernel = Exponentiation(lin_k, exponent=2)
        gpc = GaussianProcessClassifier(kernel=kernel)

        gpc.fit(self.trainX, self.trainY.ravel())
        return kernel

    def optim_params(self):
        # we use self.trainX, self.theta
        optim = self.nll_fn()
        print("optim", optim)
        res = minimize(self.nll_fn(), self.theta,
               
               method='L-BFGS-B')

        self.theta = res.x

        print(f'Optimized theta = [{self.theta[0]:.3f}], negative log likelihood = {res.fun:.3f}')
    def K_a(self, test = None):
        
        #Ka = self.init_kernel(self.trainX, self.trainX, self.theta)
        Ka = (self.trainX.to_numpy().reshape(-1, 1) @ self.trainX.to_numpy().reshape(1, -1) + self.theta**2)**self.exponent
        '''
        if test == None:

            Ka = (self.trainX.T.dot(self.trainX) + self.theta**2)**self.exponent
        else: Ka = (np.inner(self.trainX, test) + self.theta**2)**self.exponent
        print("KA shape", Ka.shape, Ka)
        '''
        return Ka


    def nll_fn(self):
        # it is needed for params function
        # Based on Algorithm 3.1 of GPML

        f = np.zeros_like(self.trainY, dtype=np.float64)
        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        for _ in range(10):
            # Line 4
            pi = expit(f)
            W = pi * (1 - pi)
            # Line 5
            W_sr = np.sqrt(W)
            W_sr_K = W_sr * self.K_a()   #[:, np.newaxis]
            prod = W_sr_K * W_sr 
            B = np.eye(W.shape[0]) + W_sr_K * W_sr 
            L = cholesky(B, lower=True)
            print("shape L, L.shape")
            # Line 6
            b = W * f + (self.trainY - pi)
            # Line 7
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            # Line 8
            f = self.K_a().dot(a)

            # Line 10: Compute log marginal likelihood in loop and use as
            #          convergence criterion
            lml = (
                -0.5 * a.T.dot(f)
                - np.log1p(np.exp(-(self.trainY * 2 - 1) * f)).sum()
                - np.log(np.diag(L)).sum()
            )
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml
        print("log_marginal_likelihood", log_marginal_likelihood)
        return log_marginal_likelihood
    

    def kernel(self, testX, derivative = False):
        # here we use self.theta, self.trainX, new testX
        
        K_a = K_a(self.trainX)
        K_s = K_a(self.trainX, testX)            
        #K_d = this_kernel.diag(self.testX)
        a_tuple = [K_a, K_s]
        return     a_tuple

    def derivate_kern(self):
        '''
        return derivative of kernel and derivative of the squared kernel
        '''

        
        if self.kernel_options == 'linear':
            
            obj1 = np.inner(self.trainX, self.testX)  
            obj2 = self.testX.T@obj1.T
            der_tuple = [self.trainX, 2*obj2.T]
        
        if self.kernel_options == 'exponential':
            der1 = self.exponent*self.trainX*self.kernel()[1]**(self.exponent - 1)
            der2= 2*self.exponent*self.trainX*self.kernel()[1]**(2*self.exponent - 1) 
            der_tuple = [der1, der2]  
        

       
            der1 = np.zeros((self.trainX))
            der2 = np.zeros((self.trainX))
            for i in self.trainX:
                der1[i] = self.exponent*self.trainX[i]*self.kernel()[1]**(self.exponent - 1)
                der2[i] = 2*self.exponent*self.trainX[i]*self.kernel()[1]**(2*self.exponent - 1) 
            der_tuple = [der1, der2]    
        
       

        return der_tuple

    def loglikelihoods(self):
        pi_, W_sr_, L_, f = self.posterior_mode(intermediate_steps = True)
        first_der = self.trainY - pi_
        sec_der = pi_ * (1 - pi_)
        a_tuple = [first_der, sec_der]
        return a_tuple
    
    def posterior_mode(self, intermediate_steps = False):
        # it is needed for params function
        # Based on Algorithm 3.1 of GPML

        f = np.zeros_like(self.trainY, dtype=np.float64)
        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        for _ in range(10):
            # Line 4
            pi = expit(f)
            W = pi * (1 - pi)
            # Line 5
            W_sr = np.sqrt(W)
            W_sr_K = W_sr * self.kernel()[0]   #[:, np.newaxis]
            prod = W_sr_K * W_sr 
            B = np.eye(W.shape[0]) + W_sr_K * W_sr 
            L = cholesky(B, lower=True)
            # Line 6
            b = W * f + (self.trainY - pi)
            # Line 7
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            # Line 8
            f = self.kernel()[0].dot(a)

            # Line 10: Compute log marginal likelihood in loop and use as
            #          convergence criterion
            lml = (
                -0.5 * a.T.dot(f)
                - np.log1p(np.exp(-(self.trainY * 2 - 1) * f)).sum()
                - np.log(np.diag(L)).sum()
            )
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        if intermediate_steps == True:
            return pi, W_sr, L, f
        else: return log_marginal_likelihood


    def params(self):
        pi_, W_sr_, L_, f = self.posterior_mode(intermediate_steps = True)
        K_star = self.kernel()[1]       
        der_loglik = self.loglikelihoods()
        post_mean = K_star.T.dot(der_loglik[0])  # Line 4
        prod = W_sr_[:, np.newaxis] * K_star
        v = solve(L_, W_sr_ * K_star)  # Line 5 #W_sr_[:, np.newaxis]
        v1 = v.T.dot(v)
        v2 = np.diag(v1)
        # Line 6 (compute np.diag(v.T.dot(v)) via einsum)
        post_var = np.ones(self.trainX.shape[0])
        
        post_var -= np.einsum("ij,ji->i", v.T, v)
        #print("post_var", self.kernel()[2].shape, post_var.shape  )
        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = post_var < 0
        if np.any(y_var_negative):
            warnings.warn(
                "Predicted variances smaller than 0. "
                "Setting those variances to 0."
            )
            post_var[y_var_negative] = 0.0
        
        a_tuple = [post_mean, post_var]
        return a_tuple
    
    def f_x (self):
        num = self.params()[0]
        den = np.sqrt(1+ np.pi * self.params()[1] / 8)
        value = num / den
        return [sigmoid(value), value]

    def FOC(self):
        sigma_x = self.f_x()[0]
        fx = self.f_x()[1]
        der_loglik = self.loglikelihoods()
        a = der_loglik[0].reshape(-1, 1) 
        W_inv = np.linalg.inv(-np.diag(der_loglik[1].ravel()))
        sum_W_K = W_inv + self.kernel()[0]
        b = np.linalg.inv(sum_W_K)
        term1 = -(np.pi /8) *self.kernel()[1].T@b
        t2 = (1/(1+(np.pi/8)*self.params()[1])).reshape(-1, 1)
        t3 = np.sqrt(self.params()[1]).reshape(-1, 1)
        t = term1* t2*t3
        der_fx = (a * (np.sqrt(t2)))*(t@self.derivate_kern()[1] + self.derivate_kern()[0])
        x = sigma_x * (der_fx + (1-sigma_x)*fx)
        return x

'''
y_obs = set_outcome(patients, rho_init, Xa_post)
X_train, X_test, y_train, y_test, Xs_train, Xs_test = sklearn.model_selection.train_test_split(torch.from_numpy(Xa_initial), 
                                                                                                torch.from_numpy(y_obs),
                                                                                                torch.from_numpy(Xs_initial))

y_train_array = y_train.detach().numpy()
x_train_array = X_train.detach().numpy().reshape(-1, 1)
x_test_array = X_test.detach().numpy().reshape(-1, 1)
df = pd.DataFrame(data={'Xa': Xa_initial,
                        'Xa_P': Xa_post,
                        'Xs': Xs_initial,
                        'states': S_prime,
                        'outcome': y_obs})
df.sort_values(by=['Xa'])
df = (df.assign(risk= lambda x: pd.cut(df['states'], 
                                                bins=[0, 0.4, 0.8, 1],
                                                labels=["L", "M", "H"])))

count_groups = df.groupby(['risk']).size().reset_index(name='counts')
indexesH = df['risk'].loc[df['risk'] == 'H'].index.tolist()
indexesM = df['risk'].loc[df['risk'] == 'M'].index.tolist()
indexesL = df['risk'].loc[df['risk'] == 'L'].index.tolist()
print("count groups", count_groups)
dfL = df.loc[indexesL]
dfM = df.loc[indexesM]
dfH = df.loc[indexesH]
arrayL_x = dfL['Xa'].to_numpy().reshape(-1, 1)
array_y = dfL['outcome'].to_numpy()
arrayX_test = np.linspace(np.min(df['Xa']), np.max(df['Xa']), num=arrayL_x.shape[0]).reshape(-1, 1)


trial = Derivate(trainX = df['Xa'].to_numpy().reshape(-1, 1), trainY = df['outcome'].to_numpy().reshape(-1, 1), testX = df['Xa_P'].to_numpy().reshape(-1, 1), specify_kernel='linear')
#fx = trial.f_x()[0]
#first_oc = trial.FOC()
posterior_mode = trial.posterior_mode()
parameters = trial.params()

'''

