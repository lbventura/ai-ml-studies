import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from image_generation_gan_arch.training_utils import (
    create_model,
    gan_checkpoint,
    gan_save_samples,
    to_var,
)
from image_generation_gan_arch.conv_utils import sample_noise
from image_generation_gan_arch.conv_utils import upconv, conv


class DCGenerator(nn.Module):  # type: ignore
    def __init__(self, noise_size: int, conv_dim: int) -> None:
        super(DCGenerator, self).__init__()

        self.conv_dim = conv_dim

        self.linear_bn = upconv(
            in_channels=noise_size,
            out_channels=conv_dim * 4,  # as given in the diagram
            kernel_size=5,
            stride=4,
            padding=2,  # see formula
            batch_norm=True,
        )
        self.upconv1 = upconv(
            in_channels=conv_dim * 4,
            out_channels=conv_dim * 2,  # as given in the diagram
            kernel_size=5,
            batch_norm=True,
        )  # this guarantees an output of 64 x 8 x 8
        self.upconv2 = upconv(
            in_channels=conv_dim * 2,
            out_channels=conv_dim,  # as given in the diagram
            kernel_size=5,
            batch_norm=True,
        )  # this guarantees an output of 32 x 16 x 16
        self.upconv3 = upconv(
            in_channels=conv_dim,
            out_channels=3,  # as given in the diagram
            kernel_size=5,
            batch_norm=False,
        )  # this guarantees an output of 3 x 32 x 32
        # note that the tanh is applied in the forward method

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generates an image given a sample of random noise.

        Input
        -----
            z: BS x noise_size x 1 x 1   -->  BSx100x1x1 (during training)

        Output
        ------
            out: BS x channels x image_width x image_height  -->  BSx3x32x32 (during training)
        """
        batch_size = z.size(0)

        out = F.relu(self.linear_bn(z)).view(
            -1, self.conv_dim * 4, 4, 4
        )  # BS x 128 x 4 x 4
        out = F.relu(self.upconv1(out))  # BS x 64 x 8 x 8
        out = F.relu(self.upconv2(out))  # BS x 32 x 16 x 16
        out = F.tanh(self.upconv3(out))  # BS x 3 x 32 x 32

        out_size = out.size()
        if out_size != torch.Size([batch_size, 3, 32, 32]):
            raise ValueError(
                "expect {} x 3 x 32 x 32, but get {}".format(batch_size, out_size)
            )
        return out


class DCDiscriminator(nn.Module):  # type: ignore
    """Defines the architecture of the discriminator network.
    Note: Both discriminators D_X and D_Y have the same architecture in this assignment.
    """

    def __init__(self, conv_dim=64) -> None:
        super(DCDiscriminator, self).__init__()

        self.kernel_size = 5
        self.stride = 2
        self.input_channels = 3

        self.conv1 = conv(
            in_channels=self.input_channels,
            out_channels=32,  # as given in the diagram
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=2,  # see formula
            batch_norm=True,
        )
        self.conv2 = conv(
            in_channels=32,
            out_channels=64,  # as given in the diagram
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=2,  # see formula
            batch_norm=True,
        )
        self.conv3 = conv(
            in_channels=64,
            out_channels=128,  # as given in the diagram
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=2,  # see formula
            batch_norm=True,
        )
        self.conv4 = conv(
            in_channels=128,
            out_channels=1,  # as given in the diagram
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=1,  # see formula, the output should be 1-dimensional
            batch_norm=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        out = F.relu(self.conv1(x))  # BS x 64 x 16 x 16
        out = F.relu(self.conv2(out))  # BS x 64 x 8 x 8
        out = F.relu(self.conv3(out))  # BS x 64 x 4 x 4

        out = self.conv4(out).squeeze()
        out_size = out.size()
        if out_size != torch.Size(
            [
                batch_size,
            ]
        ):
            raise ValueError("expect {} x 1, but get {}".format(batch_size, out_size))
        return out


def gan_training_loop(
    dataloader: DataLoader, test_dataloader: DataLoader, opts
) -> tuple[DCGenerator, DCDiscriminator]:
    """Runs the training loop.
    * Saves checkpoint every opts.checkpoint_every iterations
    * Saves generated samples every opts.sample_every iterations
    """

    # Create generators and discriminators
    G, D = create_model(opts, DCGenerator, DCDiscriminator)

    g_params = G.parameters()  # Get generator parameters
    d_params = D.parameters()  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(g_params, opts.lr, [opts.beta1, opts.beta2])
    d_optimizer = optim.Adam(d_params, opts.lr * 2.0, [opts.beta1, opts.beta2])

    train_iter = iter(dataloader)

    test_iter = iter(test_dataloader)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_noise = sample_noise(100, opts.noise_size)  # # 100 x noise_size x 1 x 1

    iter_per_epoch = len(train_iter)
    total_train_iters = opts.train_iters

    try:
        for iteration in range(1, opts.train_iters + 1):
            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                train_iter = iter(dataloader)

            real_images, real_labels = next(train_iter)
            real_images, real_labels = (
                to_var(real_images),
                to_var(real_labels).long().squeeze(),
            )

            d_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Compute the discriminator loss on real images
            D_real_loss = 0.5 * torch.mean((D(real_images) - 1) ** 2)

            # 2. Sample noise
            noise = sample_noise(100, opts.noise_size)  # # 100 x noise_size x 1 x 1

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            D_fake_loss = 0.5 * torch.mean((D(fake_images)) ** 2)

            # 5. Compute the total discriminator loss
            D_total_loss = D_real_loss + D_fake_loss

            D_total_loss.backward()
            d_optimizer.step()

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # FILL THIS IN
            # 1. Sample noise
            noise = sample_noise(100, opts.noise_size)  # # 100 x noise_size x 1 x 1

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            G_loss = torch.mean((D(fake_images) - 1) ** 2)

            G_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % opts.log_step == 0:
                print(
                    "Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}".format(
                        iteration,
                        total_train_iters,
                        D_real_loss.item(),
                        D_fake_loss.item(),
                        G_loss.item(),
                    )
                )

            # Save the generated samples
            if iteration % opts.sample_every == 0:
                gan_save_samples(G, fixed_noise, iteration, opts)

            # Save the model parameters
            if iteration % opts.checkpoint_every == 0:
                gan_checkpoint(iteration, G, D, opts)

    except KeyboardInterrupt:
        print("Exiting early from training.")
        return G, D

    return G, D
