import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm


def normal_logprob(z: Tensor, mean: Tensor, log_std: Tensor) -> Tensor:
    mean = mean + torch.tensor(0.0)
    log_std = log_std + torch.tensor(0.0)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def log_density_funnel(x: Tensor) -> Tensor:
    x1, x2 = x[:, 0], x[:, 1]
    x2_logdensity = normal_logprob(x2, torch.tensor(0.0), torch.tensor(np.log(1.35)))
    x1_logdensity = normal_logprob(x1, torch.tensor(0.0), x2)
    return x2_logdensity + x1_logdensity


def plot_isocontours(
    ax: plt.Axes,
    func: Callable,  # type: ignore
    xlimits: list[float] = [-3, 3],
    ylimits: list[float] = [-5, 3],
    numticks: int = 101,
    alpha: float = 1.0,
    cmap: plt.cm = plt.get_cmap("viridis"),
) -> None:
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    pts = torch.from_numpy(
        np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T
    )
    zs = torch.exp(func(pts)).detach().cpu().numpy()
    Z = zs.reshape(X.shape)
    ax.contour(X, Y, Z, alpha=alpha, cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)


class VAE(nn.Module):  # type: ignore
    def __init__(self, use_normalization_flow: bool = False) -> None:
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 4))
        self.decoder = nn.Sequential(nn.Linear(2, 100), nn.ReLU(), nn.Linear(100, 2))

        self.posterior_transformation = (
            nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(1, 20), nn.Tanh(), nn.Linear(20, 1))
                    for _ in range(10)
                ]
            )
            if use_normalization_flow
            else None
        )

    def transform(self, z: Tensor, transformation: nn.ModuleList) -> Tensor:
        assert transformation is not None
        for i, transform in enumerate(transformation):
            if i % 2 == 0:
                z = torch.cat([z[:, 0:1], z[:, 1:2] + transform(z[:, 0:1])], dim=1)
            else:
                z = torch.cat([z[:, 0:1] + transform(z[:, 1:2]), z[:, 1:2]], dim=1)
        return z

    def elbo(self, x: Tensor) -> tuple[Tensor, Tensor]:
        q_params = self.encoder(x)
        q_mean, q_logstd = q_params[:, :2], q_params[:, 2:]
        q_samples = torch.randn(x.shape[0], 2).to(q_mean) * torch.exp(q_logstd) + q_mean

        logqz = normal_logprob(q_samples, q_mean, q_logstd).sum(1)

        if self.posterior_transformation is not None:
            q_samples = self.transform(
                q_samples, transformation=self.posterior_transformation
            )

        logpz = normal_logprob(q_samples, torch.tensor(0.0), torch.tensor(0.0)).sum(1)
        px_mean = self.decoder(q_samples)

        logpxz = normal_logprob(x, px_mean, torch.tensor(0.0)).sum(1)

        elbo = torch.mean(logpxz + logpz - logqz)
        return elbo, q_samples


data = torch.tensor(
    [
        [5, 5],
        [0, 5],
        [5, 0],
        [-5, -5],
    ]
).float()

if __name__ == "__main__":
    plot_latent = False
    plot_final = True

    NUMBER_ITERATIONS = 10_000

    model = VAE(use_normalization_flow=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    if plot_latent:
        fig = plt.figure(figsize=(12, 3), facecolor="white")
        ax = fig.add_subplot(141, frameon=False)
        plot_isocontours(
            ax, lambda x: normal_logprob(x, torch.tensor(0.0), torch.tensor(0.0)).sum(1)
        )
        ax.set_title("Prior")
        ax.set_ylim([-3, 3])
        ax.set_xlim([-3, 3])

    q_params = model.encoder(data)
    q_mean, q_logstd = q_params[:, :2], q_params[:, 2:]

    if plot_latent:
        ax = fig.add_subplot(142, frameon=False)
        for i in range(4):
            plot_isocontours(
                ax, lambda x: normal_logprob(x.float(), q_mean[i], q_logstd[i]).sum(1)
            )
        ax.set_title("q(z|x) (Init)")
        ax.set_ylim([-3, 3])
        ax.set_xlim([-3, 3])

    for i in tqdm(range(NUMBER_ITERATIONS)):
        optimizer.zero_grad()
        elbo, q_samples = model.elbo(data)
        elbo.mul(-1).backward()
        optimizer.step()

        if i % 1000 == 0:
            print("Iteration: {}, ELBO: {:.3f}".format(i, elbo.item()))
            print("q_samples: ", q_samples)

    if plot_latent:
        ax = fig.add_subplot(143, frameon=False)
        q_params = model.encoder(data)
        q_mean, q_logstd = q_params[:, :2], q_params[:, 2:]
        for i in range(q_mean.shape[0]):
            plot_isocontours(
                ax, lambda x: normal_logprob(x.float(), q_mean[i], q_logstd[i]).sum(1)
            )
        ax.set_title("q(z|x) (After Training)")
        ax.set_ylim([-3, 3])
        ax.set_xlim([-3, 3])

        ax = fig.add_subplot(144, frameon=False)
        for i in range(q_mean.shape[0]):
            samples = (
                torch.randn(1000, 2).to(q_mean) * torch.exp(q_logstd[i][None])
                + q_mean[i][None]
            )
            ax.scatter(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy())
        ax.set_title("q(z|x) (Samples)")
        ax.set_ylim([-3, 3])
        ax.set_xlim([-3, 3])
        ax.set_yticks([])
        ax.set_xticks([])

        plt.show()

    if plot_final:
        fig_2 = plt.figure(figsize=(5, 5), facecolor="white")
        ax_2 = fig_2.add_subplot(111, frameon=False)

        for i in range(q_mean.shape[0]):
            samples = (
                torch.randn(1000, 2).to(q_mean) * torch.exp(q_logstd[i][None])
                + q_mean[i][None]
            )

            actual_space_samples = model.decoder(samples)

            ax_2.scatter(
                actual_space_samples[:, 0].detach().numpy(),
                actual_space_samples[:, 1].detach().numpy(),
                alpha=0.2,
            )
            ax_2.set_title("p(x|z) (Samples)")
            ax_2.set_ylim([-12, 12])
            ax_2.set_xlim([-12, 12])
        plt.show()

    plt.hist(actual_space_samples[:, 0].detach().numpy())
    plt.show()

    plt.hist(actual_space_samples[:, 1].detach().numpy())
    plt.show()

    print("Final ELBO: {:.3f}".format(elbo.item()))
