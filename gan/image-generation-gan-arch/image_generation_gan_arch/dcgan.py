import torch
import torch.nn.functional as F
import torch.nn as nn

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

    def __init__(self, conv_dim: int = 64) -> None:
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
