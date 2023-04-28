# evaluate a gaussian process classifier model on the dataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report, accuracy_score, log_loss


# grid search kernel for gaussian process classifier
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process.kernels import RBF, PairwiseKernel
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import WhiteKernel

import torch
import gpytorch
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
default_seed = 10000
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import UnwhitenedVariationalStrategy
from gpytorch.variational import VariationalStrategy
import seaborn as sns

# define dataset
X, y = make_classification(n_samples=100, n_features=20, n_informative=15, n_redundant=5, random_state=1)
kernel = PairwiseKernel(metric = "polynomial")
#kernel = RBF() + WhiteKernel(noise_level=0.5)

train_size = 50
X_2d_train = X[:train_size, 0:2]; y_2d_train = y[:train_size, ]; X_2d_test = X[:train_size, 0:2]; y_2d_test = y[train_size:, ]
print("Data")
print("data shape - np array", X.shape, y.shape, type(X))
print("TRAIN data - x train", X_2d_train.shape ,"y train", y_2d_train.shape)
print("TEST data - x train", X_2d_test.shape ,"y train", y_2d_test.shape)


# Create evalutaion grid
h = 0.05
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
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


print("GP sklearn -------------------------------------------------------------------------------------------------------------------")
'''
trial_fix = GaussianProcessClassifier(kernel=kernel, optimizer=None)
trial_fix.fit(X_2d_train, y_2d_train)
print("kernel n_dims", trial_fix.kernel.n_dims, "theta",  trial_fix.kernel.theta)
K_fix = trial_fix.kernel(X_2d_train)
print("K shape", K_fix.shape)
Z, objects = trial_fix.posterior_mode(K_fix, return_temporaries=True)
print("Z", Z) # this is the likelihood
pi, W_sr, L, b, a = objects
print("pi", pi.shape)
#posterior_mean_fix = trial_fix._posterior_mode(K_fix)
#print(trial_fix.f)
print(np.vstack((X_2d_test.ravel(), y_2d_test.ravel().T)).shape)
# plot the decision function for each datapoint on the grid
proba_fix = trial_fix.predict_proba(np.vstack((X_2d_test.ravel(), y_2d_test.ravel())).T)[:, 1]
proba_fix = proba_fix.reshape(X_2d_test.shape)

print("Kernel (initial): {}".format(trial_fix.kernel_))
print("Log Marginal Likelihood (initial) {}, {}".format(trial_fix.log_marginal_likelihood(trial_fix.kernel_.theta),
                                               trial_fix.log_marginal_likelihood_value_))
print("Theta (initial): %.3f"% trial_fix.kernel_.theta)
print("Accuracy: %.3f (initial)" % (accuracy_score(y_2d_train, trial_fix.predict(X_2d_train))))
print("Log-loss: %.3f (initial) "% (log_loss(y_2d_train, trial_fix.predict_proba(X_2d_train))))



trial_opt = GaussianProcessClassifier(kernel=kernel)
trial_opt.fit(X_2d_train, y_2d_train)

# plot the decision function for each datapoint on the grid
proba_optim = trial_opt.predict_proba(np.vstack((X_2d_test.ravel(), y_2d_test.ravel())).T)[:, 1]
proba_optim = proba_optim.reshape(X_2d_test.shape)

print("Kernel: {}".format(trial_opt.kernel_))
print("Log Marginal Likelihood (optimized): {}, {}".format(trial_opt.log_marginal_likelihood(trial_opt.kernel_.theta),
                                               trial_opt.log_marginal_likelihood_value_))
print("Theta (optimized): %.3f"% trial_opt.kernel_.theta)
print("Accuracy: %.3f (optimized)" % (accuracy_score(y_2d_train, trial_opt.predict(X_2d_train))))
print("Log-loss: %.3f (optimized) "% (log_loss(y_2d_train, trial_opt.predict_proba(X_2d_train))))  
'''
'''
# for 1d https://www.w3cschool.cn/doc_scikit_learn/scikit_learn-auto_examples-gaussian_process-plot_gpc.html
# Plot posteriors 
plt.figure(0)
plt.scatter(X_2d_train, y_2d_train, c='k', label="Train data")
plt.scatter(X_2d_test, y_2d_test, c='g', label="Test data")
X_ = np.linspace(0, 5, 50)
plt.plot(X_, trial_fix.predict_proba(X_[:, np.newaxis])[:, 1], 'r',
         label="Initial kernel: %s" % trial_fix.kernel_)
plt.plot(X_, trial_opt.predict_proba(X_[:, np.newaxis])[:, 1], 'b',
         label="Optimized kernel: %s" % trial_opt.kernel_)
plt.xlabel("Feature")
plt.ylabel("Class 1 probability")
plt.xlim(0, 5)
plt.ylim(-0.25, 1.5)
plt.legend(loc="best")
'''

'----------------------------------------------------------------------------------------------------------------------------------'
print("--GP GPytorch -------------------------------------------------------------------------------------------------------------------")
'''GPytorch doesn't have Laplace approx to the posterior, uses MC sampling'''


class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0)) # first dimension 
        
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        #variational_strategy = UnwhitenedVariationalStrategy(self, train_x, variational_distribution, learn_inducing_locations=False)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


# Initialize model and likelihood
model = GPClassificationModel(X_train_torch)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
training_iterations = 300

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# num_data refers to the number of training datapoints
mll = gpytorch.mlls.VariationalELBO(likelihood, 
                                    model, 
                                    y_train_torch.numel(), 
                                    combine_terms=False)
'''
for i in range(training_iterations):
    # Zero backpropped gradients from previous iteration
    optimizer.zero_grad()
    # Get predictive output
    output = model(X_train_torch)
    # Calc loss and backprop gradients
    log_lik, kl_div, log_prior = mll(output, y_train_torch)
    loss = -(log_lik - kl_div + log_prior)
    #loss = -mll(output, yt)
    loss.backward()
    
    print('Iter %d/%d - Loss: %.3f lengthscale: %.3f outputscale: %.3f' % (
        i + 1, training_iterations, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.covar_module.outputscale.item() # There is no noise in the Bernoulli likelihood
    ))
    
    optimizer.step()


# Go into eval mode
model.eval()
likelihood.eval()

with torch.no_grad():    
    test_x = torch.linspace(0, 1, 101)
    # Get classification predictions
    observed_pred = likelihood(model(X_test_torch))

    p = observed_pred.mean.numpy()
    proba_scipy = p.reshape(xx.shape)
    print(
    f"Type of output: {observed_pred.__class__.__name__}\n"
    f"Shape of output: {observed_pred.batch_shape + observed_pred.event_shape}")


# Initialize fig and axes for plot
f, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].contourf(xx,yy,proba_scipy, levels=16)
ax[1].contourf(xx,yy,proba_optim, levels=16)
ax[2].contourf(xx,yy,proba_fix, levels=16)

ax[0].scatter(X[y == 0,0], X[y == 0,1])
ax[0].scatter(X[y == 1,0], X[y == 1,1])
ax[1].scatter(X[y == 0,0], X[y == 0,1])
ax[1].scatter(X[y == 1,0], X[y == 1,1])
ax[2].scatter(X[y == 0,0], X[y == 0,1])
ax[2].scatter(X[y == 1,0], X[y == 1,1])

ax[0].set_title('GPyTorch')
ax[1].set_title('Sklearn Optim')
ax[1].set_title('Sklearn Fix')

plt.show()
'''
'--------------------------------------------------------------------------------------------------------------'
print("---another GPytorch model-----------------------------------------------------------------")
class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-1))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
def train_and_test_approximate_gp(objective_function_cls):
    model = ApproximateGPModel(torch.linspace(0, 1, 100))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    objective_function = objective_function_cls(likelihood, model, num_data=train_y.numel())
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

    # Train
    model.train()
    likelihood.train()
    for _ in range(training_iterations):
        output = model(train_x)
        loss = -objective_function(output, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Test
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        f_dist = model(train_x)
        mean = f_dist.mean
        f_lower, f_upper = f_dist.confidence_region()
        y_dist = likelihood(f_dist)
        y_lower, y_upper = y_dist.confidence_region()

    # Plot model
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    line, = ax.plot(train_x, mean, "blue")
    ax.fill_between(train_x, f_lower, f_upper, color=line.get_color(), alpha=0.3, label="q(f)")
    ax.fill_between(train_x, y_lower, y_upper, color=line.get_color(), alpha=0.1, label="p(y)")
    ax.scatter(train_x, train_y, c='k', marker='.', label="Data")
    ax.legend(loc="best")
    ax.set(xlabel="x", ylabel="y")

'--------------------------------------------------------------------------------------------------------------'
print("PLOT--------------------------------------------")
from sklearn import datasets


# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = np.array(iris.target, dtype=int)
print("data shapes", X.shape, y.shape, y)
h = 0.02  # step size in the mesh

kernel = 1.0 * RBF([1.0])
gpc_rbf_isotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)
kernel = 1.0 * RBF([1.0, 1.0])
gpc_rbf_anisotropic = GaussianProcessClassifier(kernel=kernel).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
print("data shapes for plots", x_min, x_max, y_min, y_max, x_min.shape, y_min.shape, xx.shape, yy.shape)

titles = ["Isotropic RBF", "Anisotropic RBF"]
plt.figure(figsize=(10, 5))
for i, clf in enumerate((gpc_rbf_isotropic, gpc_rbf_anisotropic)):
    # Plot the predicted probabilities. For that, we will assign a color to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.subplot(1, 2, i + 1)

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape((xx.shape[0], xx.shape[1], 3))
    print("Z shape", Z.shape)
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin="lower")

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=np.array(["r", "g", "b"])[y], edgecolors=(0, 0, 0))
    plt.xlabel("Sepal length")
    plt.ylabel("Sepal width")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(
        "%s, LML: %.3f" % (titles[i], clf.log_marginal_likelihood(clf.kernel_.theta))
    )

plt.tight_layout()
plt.show()
