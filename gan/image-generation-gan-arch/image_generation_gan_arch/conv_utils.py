import torch
import torch.nn as nn
from image_generation_gan_arch.training_utils import to_var


def sample_noise(batch_size: int, dim: int) -> torch.Tensor:
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim, 1, 1) containing uniform
      random noise in the range (-1, 1).
    """
    return to_var(torch.rand(batch_size, dim) * 2 - 1).unsqueeze(2).unsqueeze(3)


def upconv(
    in_channels, out_channels, kernel_size, stride=2, padding=2, batch_norm=True
) -> nn.Module:
    """Creates a upsample-and-convolution layer, with optional batch normalization."""
    layers = []
    if stride > 1:
        layers.append(nn.Upsample(scale_factor=stride))
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        bias=False,
    )
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 2,
    padding: int = 2,
    batch_norm: int = True,
    init_zero_weights: int = False,
) -> nn.Module:
    """Creates a convolutional layer, with optional batch normalization."""
    layers = []
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    if init_zero_weights:
        conv_layer.weight.data = (
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.001
        )
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResnetBlock(nn.Module):  # type: ignore
    def __init__(self, conv_dim) -> None:
        super(ResnetBlock, self).__init__()
        self.conv_layer = conv(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.conv_layer(x)
        return out
