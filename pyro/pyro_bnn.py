

import os
from functools import partial
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

import pyro
from pyro.distributions import Normal, Uniform, Delta, MultivariateNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.distributions.util import logsumexp
from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, TracePredictive
from pyro.infer.mcmc import MCMC, NUTS
import pyro.optim as optim
import pyro.poutine as poutine

from pyro.contrib.autoguide import AutoDiagonalNormal




# Generate random data
np.random.seed(0)
pyro.enable_validation(True)
pyro.set_rng_seed(1)
pyro.enable_validation(True)

DATASET_SIZE = 50

dict_data = {
    'cont_africa': np.random.randint(0, 2, DATASET_SIZE),
    'rugged': np.random.rand(DATASET_SIZE) * 10,
    'rgdppc_2000': np.random.rand(DATASET_SIZE) * 10
}

# Create DataFrame
data = pd.DataFrame(dict_data)

from torch.functional import F

class RegressionNN(nn.Module):
    def __init__(self, p, hidden):
        # p = number of features
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(p, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        self.factor = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        z1 = self.fc1(x)+ (self.factor * x[:, 0] * x[:, 1]).unsqueeze(1)
        h1 = torch.tanh(z1)
        z2 = self.fc2(h1) 
        
        return z2


def regression_training(x_data, y_data, loss_fn, optim, num_iterations):
    for j in range(num_iterations):
        # run the model forward on the data
        y_pred = regression_model(x_data).squeeze(-1)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.item()))
    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in regression_model.named_parameters():
        print(name, param.data.numpy())

def bayesian_model(x_data, y_data):
    # weight and bias priors
    w_prior = Normal(torch.zeros(1, 2), torch.ones(1, 2)).to_event(1)
    b_prior = Normal(torch.tensor([[8.]]), torch.tensor([[1000.]])).to_event(1)
    f_prior = Normal(0., 1.0)
    priors = {'linear.weight': w_prior, 'linear.bias': b_prior, 'factor': f_prior}
    
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    
    # sample a nn (which also samples w and b)
    lifted_reg_model = lifted_module()
    
    # Construct a conditionally independent sequence of variables. 
    scale = pyro.sample("sigma", Uniform(0., 10.))
    with pyro.plate("map", len(x_data)):
        # run the nn forward on data
        prediction_mean = lifted_reg_model(x_data).squeeze(-1)
        # condition on the observed data
        pyro.sample("obs",
                    Normal(prediction_mean, scale),
                    obs=y_data)
        return prediction_mean
    

def bayesian_regression_training(x_data, y_data, num_iterations, svi):
    pyro.clear_param_store()
    for j in range(num_iterations):
        # calculate the loss and take a gradient step
        loss = svi.step(x_data, y_data)
        if j % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss / len(x_data)))


def get_marginal(traces, sites):
    return EmpiricalMarginal(traces, sites)._get_samples_and_weights()[0].detach().cpu().numpy()


def summary(traces, sites):
    marginal = get_marginal(traces, sites)
    site_stats = {}
    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = pd.DataFrame(marginal[:, i]).transpose()
        describe = partial(pd.Series.describe, percentiles=[.05, 0.25, 0.5, 0.75, 0.95])
        site_stats[site_name] = marginal_site.apply(describe, axis=1) \
            [["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def wrapped_model(x_data, y_data):
    pyro.sample("prediction", Delta(model(x_data, y_data)))


if __name__ == '__main__':
    PLOT_DATA = False
    TRAIN_REGRESSION_MODEL = False
    # Plot the data
    if PLOT_DATA:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
        african_nations = data[data["cont_africa"] == 1]
        non_african_nations = data[data["cont_africa"] == 0]

        sns.scatterplot(x=non_african_nations["rugged"],
                    y=np.log(non_african_nations["rgdppc_2000"]),
                    ax=ax[0])
        ax[0].set(xlabel="Terrain Ruggedness Index",
                ylabel="log GDP (2000)",
                title="Non African Nations")
        sns.scatterplot(x=african_nations["rugged"],
                    y=np.log(african_nations["rgdppc_2000"]),
                    ax=ax[1])
        ax[1].set(xlabel="Terrain Ruggedness Index",
                ylabel="log GDP (2000)",
                title="African Nations")
        plt.show()

    # Create a PyTorch module for the regression model
    p = 2  # number of features
    hidden = 4 # number of hidden units
    nnet = RegressionNN(p, hidden)
    num_iterations = 2000
    input_data = torch.tensor(data.values, dtype=torch.float)
    x_data, y_data = input_data[:, :-1], input_data[:, -1]
    
    if TRAIN_REGRESSION_MODEL:

        loss_fn = torch.nn.MSELoss(reduction='sum')
        optim = torch.optim.Adam(regression_model.parameters(), lr=0.05)

        # Train the regression model
        regression_training(x_data, y_data, loss_fn, optim, num_iterations)

    # Define the bayesian model
    model = bayesian_model
    guide = AutoDiagonalNormal(model)

    optim = Adam({"lr": 0.03})
    svi = SVI(model, guide, optim, loss=Trace_ELBO(), num_samples=1000)

    # Train the Bayesian regression model
    bayesian_regression_training(x_data, y_data, num_iterations, svi)

    # Inspect learned parameters
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name))
    

    posterior = svi.run(x_data, y_data)

    # posterior predictive distribution we can get samples from
    trace_pred = TracePredictive(wrapped_model,
                                posterior,
                                num_samples=1000)
    post_pred = trace_pred.run(x_data, None)
    post_summary = summary(post_pred, sites= ['prediction', 'obs'])
    mu = post_summary["prediction"]
    y = post_summary["obs"]
    predictions = pd.DataFrame({
        "cont_africa": x_data[:, 0],
        "rugged": x_data[:, 1],
        "mu_mean": mu["mean"],
        "mu_perc_5": mu["5%"],
        "mu_perc_95": mu["95%"],
        "y_mean": y["mean"],
        "y_perc_5": y["5%"],
        "y_perc_95": y["95%"],
        "true_gdp": y_data,
    })