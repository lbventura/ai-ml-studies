import torch
import torch.nn as nn
from image_generation_gan_arch.conv_utils import conv


class ResnetBlock(nn.Module):  # type: ignore
    def __init__(self, conv_dim: int) -> None:
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
