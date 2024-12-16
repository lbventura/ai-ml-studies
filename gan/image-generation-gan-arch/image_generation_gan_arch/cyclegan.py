import torch
import torch.nn.functional as F
import torch.nn as nn

from image_generation_gan_arch.conv_utils import upconv, conv
from image_generation_gan_arch.resnet import ResnetBlock


class CycleGenerator(nn.Module):  # type: ignore
    """Defines the architecture of the generator network.
    Note: Both generators G_XtoY and G_YtoX have the same architecture in this assignment.
    """

    def __init__(self, conv_dim: int = 64, init_zero_weights: bool = False) -> None:
        super(CycleGenerator, self).__init__()

        self.kernel_size = 5
        self.stride = 2
        self.input_channels = 3

        self.conv1 = conv(
            in_channels=self.input_channels,
            out_channels=conv_dim,  # 32, as given in the diagram
            kernel_size=self.kernel_size,
            stride=self.stride,  # 2, as given in the diagram
            padding=2,  # see formula
            batch_norm=True,
            init_zero_weights=init_zero_weights,
        )
        self.conv2 = conv(
            in_channels=conv_dim,
            out_channels=conv_dim * 2,  # as given in the diagram
            kernel_size=self.kernel_size,
            stride=self.stride,  # 2, as given in the diagram
            padding=2,  # see formula
            batch_norm=True,
            init_zero_weights=init_zero_weights,
        )

        # 2. Define the transformation part of the generator
        self.resnet_block = ResnetBlock(
            conv_dim=conv_dim * 2,
        )  # the in_channels = conv_dim, and we know that the number of in_channels is equal to 64

        # 3. Define the decoder part of the generator (that builds up the output image from features)
        self.upconv1 = upconv(
            in_channels=conv_dim * 2,
            out_channels=conv_dim,  # as given in the diagram
            kernel_size=5,
            batch_norm=True,
        )  # this guarantees an output of 32 x 16 x 16
        self.upconv2 = upconv(
            in_channels=conv_dim,
            out_channels=3,  # as given in the diagram
            kernel_size=5,
            batch_norm=False,
        )  # this guarantees an output of 3 x 32 x 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates an image conditioned on an input image.

        Input
        -----
            x: BS x 3 x 32 x 32

        Output
        ------
            out: BS x 3 x 32 x 32
        """
        batch_size = x.size(0)

        out = F.relu(self.conv1(x))  # BS x 32 x 16 x 16
        out = F.relu(self.conv2(out))  # BS x 64 x 8 x 8

        out = F.relu(self.resnet_block(out))  # BS x 64 x 8 x 8

        out = F.relu(self.upconv1(out))  # BS x 32 x 16 x 16
        out = F.tanh(self.upconv2(out))  # BS x 3 x 32 x 32

        out_size = out.size()
        if out_size != torch.Size([batch_size, 3, 32, 32]):
            raise ValueError(
                "expect {} x 3 x 32 x 32, but get {}".format(batch_size, out_size)
            )

        return out
