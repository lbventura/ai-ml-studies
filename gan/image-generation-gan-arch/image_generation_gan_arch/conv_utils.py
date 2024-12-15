import torch
import torch.nn as nn


def upconv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 2,
    padding: int = 2,
    batch_norm: int = True,
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
