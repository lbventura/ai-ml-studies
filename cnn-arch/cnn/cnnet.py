import torch.nn as nn
import torch


class CNN(nn.Module):  # type: ignore
    # Define the CNN architecture
    # for colourization as a classification problem
    def __init__(
        self, kernel: int, num_filters: int, num_colours: int, num_in_channels: int
    ):
        super(CNN, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_in_channels,
                out_channels=num_filters,
                kernel_size=kernel,
                padding=padding,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.downconv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters * 2,
                kernel_size=kernel,
                padding=padding,
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.rfconv = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters * 2,
                kernel_size=kernel,
                padding=padding,
            ),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(),
        )

        self.upconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters * 2,
                out_channels=num_filters,
                kernel_size=kernel,
                padding=padding,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_colours,
                kernel_size=kernel,
                padding=padding,
            ),
            nn.BatchNorm2d(num_colours),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.finalconv = nn.Conv2d(
            in_channels=num_colours,
            out_channels=num_colours,
            kernel_size=kernel,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(self.out3)
        self.out5 = self.upconv2(self.out4)
        self.out_final = self.finalconv(self.out5)
        return self.out_final
