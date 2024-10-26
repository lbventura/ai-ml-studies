# See details https://github.com/pyro-ppl/pyro

import torch
import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist
from torch.distributions import constraints
import matplotlib.pyplot as plt

pyro.set_rng_seed(101)

def scale_parametrized_guide_constrained(guess):
    """Generates the posterior distribution for the weights given
    the guess and the measurement"""
    a = pyro.param("a", guess)
    
    b_tensor = torch.tensor(1.).clone().detach() 
    b = pyro.param("b", b_tensor, constraint=constraints.positive)
    return pyro.sample("weight", dist.Normal(a, b))  # no more torch.abs

def scale(guess):
    weight = pyro.sample("weight", dist.Normal(guess, 1.0))
    return pyro.sample("measurement", dist.Normal(weight, 0.75))



if __name__ == '__main__':
    start_optimization = True
    GUESS_VALUE = 8.5
    MEASUREMENT_VALUE = 9.5

    guess = torch.tensor(GUESS_VALUE).clone().detach() 
    conditioned_scale = pyro.condition(scale, data={"measurement": torch.tensor(MEASUREMENT_VALUE).clone().detach()})

    if start_optimization:
        pyro.clear_param_store()
        model = conditioned_scale 
        guide = scale_parametrized_guide_constrained
        svi = pyro.infer.SVI(model=model,
                            guide=guide,
                            optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                            loss=pyro.infer.Trace_ELBO())


        losses, a,b  = [], [], []
        num_steps = 2500
        for t in range(num_steps):
            losses.append(svi.step(guess))
            a.append(pyro.param("a").item())
            b.append(pyro.param("b").item())

        plt.plot(losses)
        plt.title("ELBO")
        plt.xlabel("step")
        plt.ylabel("loss")
        print('a = ',pyro.param("a").item())
        print('b = ', pyro.param("b").item())
        plt.show()

        plt.subplot(1,2,1)
        plt.plot([0,num_steps],[9.14,9.14], 'k:')
        plt.plot(a)
        plt.ylabel('a')

        plt.subplot(1,2,2)
        plt.ylabel('b')
        plt.plot([0,num_steps],[0.6,0.6], 'k:')
        plt.plot(b)
        plt.tight_layout()
        plt.show()