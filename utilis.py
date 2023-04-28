import numpy as np
from numpy import linalg as la
from scipy.linalg import cholesky, cho_solve, solve
from scipy.special import erf, expit as sigmoid

def is_pos_def(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False    





class FindMin(object):
    def __init__(
        self,
        optimized_gp
        
     
    ):
        self.optimized_gp = optimized_gp
        
        
        

    def f_x (self, test):
            
            num, var = self.optimized_gp.post_parameters(test)
            self.var = var
            den = np.sqrt(1+ np.pi * var / 8)
            
            value = np.divide(num.squeeze(), den.squeeze()) #(200,)
            
            return [sigmoid(value), value]


    def derivate_kern(self, test):
        opt_params = self.optimized_gp.kernel_.get_params()    
            
        der1 = np.zeros_like(self.Xtrain)
        der2 = np.zeros_like(self.Xtrain)
        if opt_params["metric"] == "polynomial":
            for i in range(der1.shape[0]):
                    
                    der1[i] = opt_params["gamma"]*opt_params["pairwise_kernels_kwargs"]["degree"]*self.Xtrain[i]
                    der2[i] = 2*opt_params["gamma"]*opt_params["pairwise_kernels_kwargs"]["degree"]*self.Xtrain[i]
            der11 = np.inner(((opt_params["gamma"] * np.inner(test, self.Xtrain)) **(opt_params["pairwise_kernels_kwargs"]["degree"] - 1)), der1.T)
            der22 = np.inner(((opt_params["gamma"]*np.inner(test, self.Xtrain)) **(2*opt_params["pairwise_kernels_kwargs"]["degree"] - 1)), der2.T)
            der_tuple = [der11, der22] # shape of each is n,self.opt_params
        

        return der_tuple
    def FOC_i(self, test):
        # computes derivative function
            
        K_opt = self.optimized_gp.kernel(self.Xtrain)
        Z_c_opt, objects_c_opt = self.optimized_gp.posterior_mode(K_opt, return_temporaries=True)
        pi_c_opt, W_sr_c_opt, L_c_opt, b_c_opt, a_c_opt = objects_c_opt

        values_for_min = np.zeros((self.Xtrain.shape[0],self.Xtrain.shape[1]))
        for i in range(values_for_min.shape[0]):
            sigma_x = self.f_x(test)[0][i] # now scalar ()
            fx = self.f_x(test)[1][i] 
            der2_loglik = (pi_c_opt * (1 - pi_c_opt))[i]
            W_inv = - (1/der2_loglik)
            sum_W_K = W_inv + self.optimized_gp.kernel(test[i])
            b = 1/sum_W_K
            a =  (self.Ytrain[i] - pi_c_opt[i]).reshape(-1, 1) 
            t1 = -(np.pi /8) *self.optimized_gp.kernel(test[i])*b
            t2 = (1/(1+(np.pi/8)*self.var[i])).reshape(-1, 1)     
            t3 = np.sqrt(self.var[i]).reshape(-1, 1)
            t = t1* t2*t3
            der_fx = (a * (np.sqrt(t2)))*(t*self.derivate_kern(test)[1][i] + self.derivate_kern(test)[1][i])
            x = sigma_x * (der_fx.squeeze() + (1-sigma_x)*fx)
            values_for_min[i]=x
        return values_for_min
    
    def search_Xa(self, Xtrain, Ytrain):
        self.Ytrain = Ytrain
        self.Xtrain = Xtrain    
        values = np.zeros_like(self.Ytrain.reshape(-1,1))        
        i = 0
        for i in range(5):
            print("df input", self.Xtrain.shape)
            trial = np.random.uniform(-4, 4, self.Xtrain.shape[0]) # always n, 1
            
            trial_df = np.vstack([self.Xtrain[: , :-1], trial.reshape(-1, 1)]).reshape(-1, self.Xtrain.shape[1]) # Xs, Xa_star
            print("trial_df", trial_df.shape)
            #values = np.column_stack((values, trial))
            b = self.FOC_i(trial_df).squeeze()
            print("b", b.shape)
            b_1 = b[:, -1]   
            values = np.column_stack((values,b_1))
            i += 1
        values = np.delete(values, 0,1)
    
        return np.array(values).min(axis=1) # shape (n,)