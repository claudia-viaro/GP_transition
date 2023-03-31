import torch
import numpy as np
import os
import copy
import sklearn
import warnings
from numbers import Integral
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, expit as sigmoid
import pandas as pd
from sklearn.gaussian_process.kernels import PairwiseKernel, RBF, ConstantKernel as C
from scipy.special import erf, expit
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.base import clone 
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import (RationalQuadratic,Exponentiation)
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from utils import is_pos_def

class GaussianProcessClassifierLaplace(object):
    '''
    Parameters
    ----------
    kernel : kernel instance, default=None
        The kernel specifying the covariance function of the GP. If None is
        passed, the kernel "1.0 * RBF(1.0)" is used as default. Note that
        the kernel's hyperparameters are optimized during fitting.
    optimizer : 'fmin_l_bfgs_b' or callable, default='fmin_l_bfgs_b'
    n_restarts_optimizer : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that n_restarts_optimizer=0 implies that one
        run is performed.
    max_iter_predict : int, default=100
        The maximum number of iterations in Newton's method for approximating
        the posterior during predict. Smaller values will reduce computation
        time at the cost of worse results.
    warm_start : bool, default=False
        If warm-starts are enabled, the solution of the last Newton iteration
        on the Laplace approximation of the posterior mode is used as
        initialization for the next call of _posterior_mode(). This can speed
        up convergence when _posterior_mode is called several times on similar
        problems as in hyperparameter optimization. See :term:`the Glossary
        <warm_start>`.
    copy_X_train : bool, default=True
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation used to initialize the centers.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
    Attributes
    ----------
    X_train_ : array-like of shape (n_samples, n_features) or list of object
        Feature vectors or other representations of training data (also
        required for prediction).
    y_train_ : array-like of shape (n_samples,)
        Target values in training data (also required for prediction)
    classes_ : array-like of shape (n_classes,)
        Unique class labels.
    kernel_ : kernl instance
        The kernel used for prediction. The structure of the kernel is the
        same as the one passed as parameter but with optimized hyperparameters
    L_ : array-like of shape (n_samples, n_samples)
        Lower-triangular Cholesky decomposition of the kernel in X_train_
    pi_ : array-like of shape (n_samples,)
        The probabilities of the positive class for the training points
        X_train_
    W_sr_ : array-like of shape (n_samples,)
        Square root of W, the Hessian of log-likelihood of the latent function
        values for the observed labels. Since W is diagonal, only the diagonal
        of sqrt(W) is stored.
    log_marginal_likelihood_value_ : float
        The log-marginal-likelihood of ``self.kernel_.theta``
    References
    ----------
    .. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
       "Gaussian Processes for Machine Learning",
       MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_
    '''

    def __init__(
        self,
        kernel=None,
        kernel_options = "rbf",
        *,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=200, #100
        warm_start=False,
        copy_X_train=True
    ):
        self.kernel = kernel
        self.kernel_options = kernel_options
        self.optimizer = optimizer
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.copy_X_train = copy_X_train

        self.Xa_post_vector = []
        self.Jmin_vector_value_ = []
        

    def fit(self, X, y):
        """Fit Gaussian process classification model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of object
            Feature vectors or other representations of training data.
        y : array-like of shape (n_samples,)
            Target values, must be binary.
        Returns
        -------
        self : returns an instance of self.
        """
        self.kernel_ = clone(self.kernel)

        self.rng = np.random.mtrand._rand

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = y
        # Encode class labels and check that it is a binary classification
        # problem
        
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False
                    )
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta, clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [
                self._constrained_optimization(
                    obj_func, self.kernel_.theta, self.kernel_.bounds
                )
            ]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = np.exp(self.rng.uniform(bounds[:, 0], bounds[:, 1]))
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()
            
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # K = self.kernel_(self.X_train_)

        
        return self

    def f_x_i(self, Xtest):
        # Xtest is a single instance of the vector newXa
        num, var = self.post_parameters(Xtest)
    
        den = np.sqrt(1+ np.pi * var / 8)
        
        value = np.divide(num.squeeze(), den.squeeze()) #(200,)
        
        return sigmoid(value), value

    def J_function(self, newXa):
        # Compute function J
        J, other = self.f_x_i(newXa)

        # Compute gradient 
        grad = self.obj_FOC_i

        return J, grad
        
    

    def log_marginal_likelihood(
        self, theta=None, eval_gradient=False, clone_kernel=True
    ):
        """Returns log-marginal likelihood of theta for training data.
        Parameters
        ----------
        theta : array-like of shape (n_kernel_params,), default=None
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated. If None, the precomputed log_marginal_likelihood
            of ``self.kernel_.theta`` is returned.
        eval_gradient : bool, default=False
            If True, the gradient of the log-marginal likelihood with respect
            to the kernel hyperparameters at position theta is returned
            additionally. If True, theta must not be None.
        clone_kernel : bool, default=True
            If True, the kernel attribute is copied. If False, the kernel
            attribute is modified, but may result in a performance improvement.
        Returns
        -------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        log_likelihood_gradient : ndarray of shape (n_kernel_params,), \
                optional
            Gradient of the log-marginal likelihood with respect to the kernel
            hyperparameters at position theta.
            Only returned when `eval_gradient` is True.
        """
        if theta is None:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta

        if eval_gradient:
            K, K_gradient = kernel(self.X_train_, eval_gradient=True)
        else:
            K = kernel(self.X_train_)

        # Compute log-marginal-likelihood Z and also store some temporaries
        # which can be reused for computing Z's gradient
        
        Z, pi, W_sr, L, b, a, f = self._posterior_mode(K)
        if not eval_gradient:
            return Z

        # Compute gradient based on Algorithm 5.1 of GPML
        d_Z = np.empty(theta.shape[0])
        # XXX: Get rid of the np.diag() in the next line
        R = W_sr.T * cho_solve((L, True), W_sr)  # Line 7
        W_sr_K = W_sr * K
        C = solve(L, W_sr * K)  # Line 8
        # Line 9: (use einsum to compute np.diag(C.T.dot(C))))
        s_2 = (
            -0.5
            * (np.diag(K) - np.einsum("ij, ij -> j", C, C)).reshape(-1,1)
            * (pi * (1 - pi) * (1 - 2 * pi))
        )  # third derivative
        for j in range(d_Z.shape[0]):
            C = K_gradient[:, :, j]  # Line 11
            # Line 12: (R.T.ravel().dot(C.ravel()) = np.trace(R.dot(C)))
            
            s_1 = 0.5 * a.T.dot(C).dot(a) - 0.5 * R.T.ravel().dot(C.ravel())
            b = C.dot(self.y_train_ - pi)  # Line 13
            s_3 =b - K.dot(R.T.dot(b))  # Line 14  # b - K.dot(R*b)  #
            d_Z[j] = s_1 + s_2.T.dot(s_3)  # Line 15
        # returns loglik and derivatives?
        return Z, d_Z

    def _posterior_mode(self, K):
        """Mode-finding for binary Laplace GPC and fixed kernel.
        This approximates the posterior of the latent function values for given
        inputs and target observations with a Gaussian approximation and uses
        Newton's iteration to find the mode of this approximation.
        """
        # Based on Algorithm 3.1 of GPML

        # If warm_start are enabled, we reuse the last solution for the
        # posterior mode as initialization; otherwise, we initialize with 0
        if (
            self.warm_start
            and hasattr(self, "f_cached")
            and self.f_cached.shape == (self.y_train_.shape, 1)
        ):
            f = self.f_cached
        else:
            f = np.zeros_like(self.y_train_, dtype=np.float64).reshape(-1, 1)
        # Use Newton's iteration method to find mode of Laplace approximation
        log_marginal_likelihood = -np.inf
        for _ in range(self.max_iter_predict):
            # Line 4
            pi = expit(f)
            W = pi * (1 - pi)
            # Line 5
            W_sr = np.sqrt(W)
            W_sr_K = W_sr * K
            #W_sr_K = W_sr[:, np.newaxis] * K

            self.B = np.eye(W.shape[0]) + W_sr_K * W_sr
            L = cholesky(self.B, lower=True)
            # Line 6
            b = W * f + (self.y_train_ - pi)
            # Line 7
            cho = cho_solve((L, True), W_sr_K.dot(b))
            cho_prod = W_sr * cho
            a = b - W_sr * cho_solve((L, True), W_sr_K.dot(b))
            # Line 8
            f = K.dot(a)

            # Line 10: Compute log marginal likelihood in loop and use as
            #          convergence criterion
            lml = (
                -0.5 * a.T.dot(f)
                - np.log1p(np.exp(-(self.y_train_ * 2 - 1) * f)).sum()
                - np.log(np.diag(L)).sum()
            )
            # Check if we have converged (log marginal likelihood does
            # not decrease)
            # XXX: more complex convergence criterion
            if lml - log_marginal_likelihood < 1e-10:
                break
            log_marginal_likelihood = lml

        self.f_cached = f  # Remember solution for later warm-starts
        self.pi = pi
        self.L = L
        self.f = f
        self.W_sr = W_sr
        self.a = a
        return log_marginal_likelihood, pi, W_sr, L, b, a, f


    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
    
    def post_parameters(self, X):
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
        K_star = self.kernel_(self.X_train_, X)  # K_star =k(x_star)
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
    
    def f_x (self, Xtest):
        num, var = self.post_parameters(Xtest)
    
        den = np.sqrt(1+ np.pi * var / 8)
        
        value = np.divide(num.squeeze(), den.squeeze()) #(200,)
        
        return [sigmoid(value), value]
    

    def fit_Xa(self):
        
        newXA_bounds = np.array([-4, 4])
        newXA_bounds = np.hstack(newXA_bounds, newXA_bounds)

        
        def obj_func(newXA):
            J_fun, grad = self.J_function(newXA)
            return -J_fun, -grad
            
        # set object to collect optimal values
        optima = []
        
        # Additional runs are performed from log-uniform chosen initial
        # newXA
        
        iteration = 0
        for iteration in range(10):
            Xa_initial = np.exp(self.rng.uniform(newXA_bounds[:,0], newXA_bounds[:,1]))
            print("Xa_initial", Xa_initial)
            optima.append(
                self._constrained_optimization(obj_func, Xa_initial, newXA_bounds)
            )
            iteration += 1
        print(optima)
        # Select result from run with minimal (negative) J
        J_values = list(map(itemgetter(1), optima))
        self.Xa_post_vector.append(optima[np.argmin(J_values)][0])
        self.Jmin_vector_value_.append(-np.min(J_values))


        # Precompute quantities required for predictions which are independent
        # of actual query points
        # K = self.kernel_(self.X_train_)

        
        return optima[np.argmin(J_values)][0]

    
    def derivate_kern(self, testX):
        '''
        return derivative of kernel and derivative of the squared kernel
        '''
        
        
        if self.kernel_options == 'linear':
            
            obj1 = np.inner(self.X_train_, testX)  
            obj2 = self.X_train_.T@obj1.T
            der_tuple = [self.X_train_, 2*obj2.T]
        
        if self.kernel_options == 'exponential':
            der1 = self.exponent*self.X_train_*self.kernel()[1]**(self.exponent - 1)
            der2= 2*self.exponent*self.X_train_*self.kernel()[1]**(2*self.exponent - 1) 
            der_tuple = [der1, der2]  
        
        
       
            der1 = [0]*self.X_train_.shape[0]
            der2 = [0]*self.X_train_.shape[0]
            for i in self.trainX:
                der1[i] = self.exponent*self.X_train_[i]*self.kernel()[1]**(self.exponent - 1)
                der2[i] = 2*self.exponent*self.X_train_[i]*self.kernel()[1]**(2*self.exponent - 1) 
            der_tuple = [der1, der2]    
        if self.kernel_options == 'rbf':
            
            der1 = [0]*self.X_train_.shape[0]
            der2 = [0]*self.X_train_.shape[0]
            gamma = 1/(2*(self.kernel_.theta[1]**2))
            
            
            for i in range(self.X_train_.shape[0]):
                
                distance = self.X_train_[i][0]-testX[i][0]
                der1[i] = -2*gamma*distance*self.kernel(self.X_train_[i])
                der2[i] = -4*gamma*distance*self.kernel(testX[i])

            der_tuple = [np.array(der1).squeeze(), np.array(der2).squeeze()] 

       

        return der_tuple

    def derivate_kern_i(self, testX):
        '''
        return derivative of kernel and derivative of the squared kernel
        testX is a single instance of the vector
        '''
        
        
           
        if self.kernel_options == 'rbf':
            
            
            gamma = 1/(2*(self.kernel_.theta[1]**2))
            
            
            
                
            distance = self.X_train_[0][0]-testX[0][0]
            der1 = -2*gamma*distance*self.kernel(self.X_train_[0])
            der2 = -4*gamma*distance*self.kernel(testX)

            der_tuple = [np.array(der1).squeeze(), np.array(der2).squeeze()] 

       

        return der_tuple
    

    def FOC_i(self, Xtest):
        # computes derivative function
        
        
        values = [0]*Xtest.shape[0]
        for i in range(Xtest.shape[0]):

            sigma_x = self.f_x(Xtest)[1][i] # now scalar ()
            
            fx = self.f_x(Xtest)[0][i] 
            der2_loglik = (self.pi * (1 - self.pi))[i]
            W_inv = - (1/der2_loglik)
            sum_W_K = W_inv + self.kernel(Xtest[i])
            b = 1/sum_W_K

            a =  (self.y_train_[i] - self.pi[i]).reshape(-1, 1)  
            t1 = -(np.pi /8) *self.kernel(Xtest[i])*b
            t2 = (1/(1+(np.pi/8)*self.post_parameters(Xtest)[1][i])).reshape(-1, 1)            
            t3 = np.sqrt(self.post_parameters(Xtest)[1][i]).reshape(-1, 1)
            t = t1* t2*t3
            der_fx = (a * (np.sqrt(t2)))*(t*self.derivate_kern(Xtest)[1][i] + self.derivate_kern(Xtest)[0][i])
            
            x = sigma_x * (der_fx.squeeze() + (1-sigma_x)*fx)
            values[i]=x
        return np.array(values)
    
    def obj_FOC_i(self, Xtest):
    # Xtest is a single instance of the vector
        
        
        

        sigma_x = self.f_x_i(Xtest)[1] # now scalar ()

        
        fx = self.f_x(Xtest)[0]
        der2_loglik = self.pi[0] * (1 - self.pi[0])
        W_inv = - (1/der2_loglik)
        sum_W_K = W_inv + self.kernel(Xtest)
        b = 1/sum_W_K

        a =  (self.y_train_[0] - self.pi) .reshape(-1, 1)  
        t1 = -(np.pi /8) *self.kernel(Xtest)*b
        t2 = (1/(1+(np.pi/8)*self.post_parameters(Xtest)[1])).reshape(-1, 1)            
        t3 = np.sqrt(self.post_parameters(Xtest)[1]).reshape(-1, 1)
        t = t1* t2*t3
        der_fx = (a * (np.sqrt(t2)))*(t*self.derivate_kern(Xtest)[1] + self.derivate_kern(Xtest)[0])
        
        x = sigma_x * (der_fx.squeeze() + (1-sigma_x)*fx)        
        return x

    def FOC(self, Xtest):
        # computes derivative
        sigma_x = self.f_x(Xtest)[1]
        fx = self.f_x(Xtest)[0] 
        der_loglik = self.log_marginal_likelihood_value_ #constant

        der2_loglik = self.pi * (1 - self.pi) #vector

        a = der_loglik.reshape(-1, 1) 
        W_inv = np.linalg.inv(-np.diag(der2_loglik.ravel()))
        print("shape w inv", W_inv.shape)
        sum_W_K = W_inv + self.kernel(Xtest)
        b = np.linalg.inv(sum_W_K)
        print("shape b", b.shape)
        term1 = -(np.pi /8) *self.kernel(Xtest).T@b
        t2 = (1/(1+(np.pi/8)*self.post_parameters(Xtest)[1])).reshape(-1, 1)

        t3 = np.sqrt(self.post_parameters(Xtest)[1]).reshape(-1, 1)
        t = term1* t2*t3
        print("t", t.shape)
        der_fx = (a * (np.sqrt(t2)))*(t@self.derivate_kern(Xtest)[1] + self.derivate_kern(Xtest)[0])
        x = sigma_x * (der_fx + (1-sigma_x)*fx)
        print("der_fx", der_fx.shape)
        return x
    
    def find_min(self):
         
        values = np.zeros_like(self.y_train_.reshape(-1,1))
        
        i = 0
        for i in range(20):
            trial = np.random.uniform(-4, 4,200)
            #values = np.column_stack((values, trial))
            b = self.FOC_i(trial.reshape(-1, 1)).squeeze()

            values = np.column_stack((values,b))
            i += 1
        values = np.delete(values, 0,1)
    
        return np.array(values).min(axis=1)

def _check_optimize_result(solver, result, max_iter=None, extra_warning_msg=None):
    """Check the OptimizeResult for successful convergence
    Parameters
    ----------
    solver : str
       Solver name. Currently only `lbfgs` is supported.
    result : OptimizeResult
       Result of the scipy.optimize.minimize function.
    max_iter : int, default=None
       Expected maximum number of iterations.
    extra_warning_msg : str, default=None
        Extra warning message.
    Returns
    -------
    n_iter : int
       Number of iterations.
    """
    # handle both scipy and scikit-learn solver names
    if solver == "lbfgs":
        if result.status != 0:
            try:
                # The message is already decoded in scipy>=1.6.0
                result_message = result.message.decode("latin1")
            except AttributeError:
                result_message = result.message
            warning_msg = (
                "{} failed to converge (status={}):\n{}.\n\n"
                "Increase the number of iterations (max_iter) "
                "or scale the data as shown in:\n"
                "    https://scikit-learn.org/stable/modules/"
                "preprocessing.html"
            ).format(solver, result.status, result_message)
            if extra_warning_msg is not None:
                warning_msg += "\n" + extra_warning_msg
            warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
        if max_iter is not None:
            # In scipy <= 1.0.0, nit may exceed maxiter for lbfgs.
            # See https://github.com/scipy/scipy/issues/7854
            n_iter_i = min(result.nit, max_iter)
        else:
            n_iter_i = result.nit
    else:
        raise NotImplementedError

    return n_iter_i

    

__all__ = [
    "NotFittedError",
    "ConvergenceWarning",
    "DataConversionWarning",
    "DataDimensionalityWarning",
    "EfficiencyWarning",
    "FitFailedWarning",
    "SkipTestWarning",
    "UndefinedMetricWarning",
    "PositiveSpectrumWarning",
]


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.
    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.
    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    NotFittedError("This LinearSVC instance is not fitted yet. Call 'fit' with
    appropriate arguments before using this estimator."...)
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.
    """


class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems
    .. versionchanged:: 0.18
       Moved from sklearn.utils.
    """


