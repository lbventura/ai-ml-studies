import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, kernel, num_filters, num_colours, num_in_channels):
        super(CNN, self).__init__()
        padding = kernel // 2

        self.downconv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_in_channels,
                out_channels=num_filters,
                kernel_size=kernel,
                padding=padding,
            ),  # input size is now num_in_channels
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
            ),  # output size is now number of colors
            nn.BatchNorm2d(num_colours),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )
        self.finalconv = nn.Conv2d(
            in_channels=num_colours,
            out_channels=num_colours,
            kernel_size=kernel,
            padding=padding,
        )  # Change the output size from 3 (RBG) to the number of colors

    def forward(self, x):
        self.out1 = self.downconv1(x)
        self.out2 = self.downconv2(self.out1)
        self.out3 = self.rfconv(self.out2)
        self.out4 = self.upconv1(self.out3)
        self.out5 = self.upconv2(self.out4)
        self.out_final = self.finalconv(self.out5)
        return self.out_final
