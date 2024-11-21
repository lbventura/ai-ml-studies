"""
This is an adaptation of the GAN implementation from the following repository:
https://github.com/AYLIEN/gan-intro/tree/master
"""

from dataclasses import dataclass
from enum import StrEnum

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import ks_2samp

sns.set_theme(color_codes=True)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class LossChoice(StrEnum):
    BCE = "BCE"
    MSE = "MSE"


@dataclass
class ExecutionParams:
    represent_real_data: bool = True  # represent real data
    num_steps: int = 5000  # the number of training steps to take
    hidden_size: int = 4  # 'MLP hidden size'
    batch_size: int = 8  # the batch size
    learning_rate: float = 0.001  # the learning rate
    gen_number_extra_layers: int = 0  # number of intermediate layers in the generator
    disc_number_intermediate_layers: int = (
        1  # number of intermediate layers in the discriminator
    )
    minibatch_disc: bool = False  # use minibatch discrimination
    bimodal: bool = False  # use bimodal data distribution
    loss_choice: LossChoice = LossChoice.BCE  # the choice of loss function

    def __post_init__(self) -> None:
        if self.disc_number_intermediate_layers < 1:
            raise ValueError(
                "The number of intermediate layers in the discriminator must be greater than 0"
            )

        if self.gen_number_extra_layers < 0:
            raise ValueError(
                "The number of extra layers in the generator must be non-negative"
            )


class DataDistribution:
    def __init__(self, bimodal: bool = False):
        self.mu = 4
        self.sigma = 0.5
        self.bimodal = bimodal

    def sample(self, N: int) -> np.ndarray:
        number_samples = N // 2 if self.bimodal else N
        samples_mode_1 = np.random.normal(self.mu, self.sigma, number_samples)
        samples_mode_2 = (
            np.random.normal(self.mu / 2, self.sigma / 2, number_samples)
            if self.bimodal
            else []
        )
        samples = np.concatenate((samples_mode_1, samples_mode_2))
        samples.sort()
        return samples


class GeneratorDistribution:
    def __init__(self, value_range: int) -> None:
        self.range = value_range

    def sample(self, N: int) -> np.ndarray:
        # stratified sampling
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01


class Generator(nn.Module):  # type: ignore
    def __init__(
        self, input_dim: int, hidden_dim: int, gen_number_extra_layers: int = 0
    ):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.intermediate_layers = (
            nn.Sequential(
                *[
                    layer
                    for _ in range(gen_number_extra_layers)
                    for layer in (nn.Linear(hidden_dim, hidden_dim), nn.Softplus())
                ]
            )
            if gen_number_extra_layers > 0
            else None
        )
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # the softplus is a smooth approximation of the ReLU function
        h0 = torch.nn.functional.softplus(self.fc1(x))

        if self.intermediate_layers:
            h0 = self.intermediate_layers(h0)
        h1 = self.fc2(h0)
        return h1


class Discriminator(nn.Module):  # type: ignore
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        disc_number_intermediate_layers: int = 1,
        minibatch_layer: bool = True,
    ):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(hidden_dim * 2, hidden_dim * 2)
                for _ in range(disc_number_intermediate_layers)
            ]
        )
        self.minibatch_layer = (
            MinibatchDiscrimination(
                input_dim=hidden_dim * 2, num_kernels=5, kernel_dim=3
            )
            if minibatch_layer
            else None
        )
        self.fc3 = nn.Linear(
            hidden_dim * 2 + (5 if minibatch_layer else 0), 1
        )  # this is because the minibatch layer adds 5 features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.nn.functional.relu(self.fc1(x))
        h1 = torch.nn.functional.relu(self.fc2(h0))
        if self.minibatch_layer:
            h2 = self.minibatch_layer(h1)
        else:
            h2 = h1
        h3 = torch.sigmoid(self.fc3(h2))
        return h3


class MinibatchDiscrimination(nn.Module):  # type: ignore
    """
    Create a set of new features (activations) that capture relationships between the input samples.
    These new features can then be used to perform minibatch discrimination, which helps the discriminator in a GAN
    to detect correlations between samples in a minibatch. This can improve the discriminator's ability to distinguish between real and fake samples,
    especially when the generator tries to produce similar samples within a minibatch.
    """

    def __init__(self, input_dim: int, num_kernels: int, kernel_dim: int):
        # note that this implicitly introduces more parameters in the discriminator
        super(MinibatchDiscrimination, self).__init__()
        self.T = nn.Parameter(torch.Tensor(input_dim, num_kernels * kernel_dim))
        nn.init.normal_(self.T, mean=0, std=0.02)
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activation = x.mm(self.T).view(-1, self.num_kernels, self.kernel_dim)
        diffs = activation.unsqueeze(3) - activation.permute(1, 2, 0).unsqueeze(0)
        abs_diffs = torch.sum(torch.abs(diffs), 2)
        minibatch_features = torch.sum(torch.exp(-abs_diffs), 2)
        return torch.cat([x, minibatch_features], 1)


class GAN:
    def __init__(self, params: ExecutionParams):
        self.G = Generator(
            1,
            hidden_dim=params.hidden_size,
            gen_number_extra_layers=params.gen_number_extra_layers,
        )
        self.D = Discriminator(
            1,
            hidden_dim=params.hidden_size,
            disc_number_intermediate_layers=params.disc_number_intermediate_layers,
            minibatch_layer=params.minibatch_disc,
        )

        match params.loss_choice:
            case LossChoice.BCE:
                self.loss_d = nn.BCELoss()
                self.loss_g = nn.BCELoss()
            case LossChoice.MSE:
                self.loss_d = nn.MSELoss()
                self.loss_g = nn.MSELoss()
            case _:
                raise ValueError("Invalid loss choice")

        self.opt_d = optim.Adam(self.D.parameters(), lr=params.learning_rate)
        self.opt_g = optim.Adam(self.G.parameters(), lr=params.learning_rate)

    def train_discriminator(
        self, real_data: torch.Tensor, fake_data: torch.Tensor
    ) -> torch.Tensor:
        self.opt_d.zero_grad()

        prediction_real = self.D(real_data)
        error_real = self.loss_d(prediction_real, torch.ones_like(prediction_real))

        prediction_fake = self.D(fake_data)
        error_fake = self.loss_d(prediction_fake, torch.zeros_like(prediction_fake))

        total_error = error_real + error_fake
        total_error.backward()

        self.opt_d.step()

        return total_error

    def train_generator(self, fake_data: torch.Tensor) -> torch.Tensor:
        self.opt_g.zero_grad()

        prediction = self.D(fake_data)
        error = self.loss_g(prediction, torch.ones_like(prediction))
        error.backward()

        self.opt_g.step()

        return error


def train(
    model: GAN,
    data: DataDistribution,
    gen: GeneratorDistribution,
    params: ExecutionParams,
) -> None:
    print("Step: D loss\t G loss")

    for step in range(params.num_steps + 1):
        # update discriminator
        x = torch.Tensor(data.sample(params.batch_size)).view(params.batch_size, 1)
        z = torch.Tensor(gen.sample(params.batch_size)).view(params.batch_size, 1)
        fake_data = model.G(z)
        loss_d = model.train_discriminator(x, fake_data)

        # update generator
        z = torch.Tensor(gen.sample(params.batch_size)).view(params.batch_size, 1)
        fake_data = model.G(z)
        loss_g = model.train_generator(fake_data)

        if step % 100 == 0:
            print(f"{step}: {loss_d.item():.4f}\t{loss_g.item():.4f}")

    # plot the final distributions
    (
        decision_boundary,
        data_distribution,
        generator_distribution,
        ks_statistic,
        p_value,
    ) = samples(
        model=model, data=data, sample_range=gen.range, batch_size=params.batch_size
    )
    plot_distributions(
        decision_boundary=decision_boundary,
        data_distribution=data_distribution,
        generator_distribution=generator_distribution,
        sample_range=gen.range,
    )

    # print the Kolmogorov-Smirnov test results
    print("KS Statistic: ", ks_statistic)
    print("P-Value: ", p_value)


def samples(
    model: GAN,
    data: DataDistribution,
    sample_range: int,
    batch_size: int,
    num_points: int = 10000,
    num_bins: int = 200,
) -> tuple[np.array, np.array, np.array, np.array, np.array]:
    """
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    """

    def batch_processor(
        sample_range: int,
        num_points: int,
        batch_size: int,
        input_function: Discriminator | Generator,
    ) -> np.array:
        l_space = np.linspace(-sample_range, sample_range, num_points)
        number_batches = num_points // batch_size
        c_tensor = torch.Tensor(l_space).view(number_batches, batch_size, 1)

        output_array = np.zeros((num_points, 1))
        for i in range(number_batches):
            batch_tensor = c_tensor[i]
            output_array[batch_size * i : batch_size * (i + 1)] = (
                input_function(batch_tensor).detach().numpy()
            )
        return output_array

    db = batch_processor(
        sample_range=sample_range,
        num_points=num_points,
        batch_size=batch_size,
        input_function=model.D,
    )

    # data distribution
    d = data.sample(num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)
    pd, _ = np.histogram(d, bins=bins, density=True)

    g = batch_processor(
        sample_range=sample_range,
        num_points=num_points,
        batch_size=batch_size,
        input_function=model.G,
    )

    pg, _ = np.histogram(g, bins=bins, density=True)

    # compute the Kolmogorov-Smirnov test
    ks_statistic, p_value = ks_2samp(pd, pg)

    return db, pd, pg, ks_statistic, p_value


def plot_distributions(
    decision_boundary: np.array,
    data_distribution: np.array,
    generator_distribution: np.array,
    sample_range: int,
) -> None:
    db_x = np.linspace(-sample_range, sample_range, len(decision_boundary))
    p_x = np.linspace(-sample_range, sample_range, len(data_distribution))

    _, ax = plt.subplots(1)
    ax.plot(db_x, decision_boundary, label="Decision boundary")
    ax.set_ylim(0, 1)

    plt.plot(p_x, data_distribution, label="Real data")
    plt.plot(p_x, generator_distribution, label="Generated data")

    plt.title("1D Generative Adversarial Network")
    plt.xlabel("Data values")
    plt.ylabel("Probability density")
    plt.legend()
    plt.show()


def main(params: ExecutionParams) -> None:
    if params.represent_real_data:
        samples = 10000
        sample_data = DataDistribution(bimodal=params.bimodal).sample(samples)

        data_dist, x_data_dist = np.histogram(sample_data, bins=200, density=True)

        plt.plot(x_data_dist[:-1], data_dist, label="Real data")
        plt.show()

    model = GAN(params=params)
    train(
        model,
        DataDistribution(bimodal=params.bimodal),
        GeneratorDistribution(value_range=8),
        params=params,
    )


if __name__ == "__main__":
    params = ExecutionParams(
        num_steps=10_000,
        disc_number_intermediate_layers=1,
        gen_number_extra_layers=1,
        minibatch_disc=True,
        bimodal=True,
    )
    main(params=params)
