import torch
import torch.nn as nn

## Discriminator

first_conv_layer = nn.Conv2d(
    in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2
)

# Create a sample input tensor
input_tensor = torch.rand(3, 32, 32)

# Pass the input tensor through the convolutional layer
output_tensor = first_conv_layer(input_tensor)

# Print the input and output tensors
print("Input Tensor:")
print(input_tensor.shape)
print("\nOutput Tensor:")
print(output_tensor.shape)

assert first_conv_layer(torch.rand(3, 32, 32)).shape == torch.Size([32, 16, 16])

second_conv_layer = nn.Conv2d(
    in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2
)
assert second_conv_layer(torch.rand(32, 16, 16)).shape == torch.Size([64, 8, 8])

second_conv_layer = nn.Conv2d(
    in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2
)
assert second_conv_layer(torch.rand(64, 8, 8)).shape == torch.Size([128, 4, 4])

last_conv_layer = nn.Conv2d(
    in_channels=128, out_channels=1, kernel_size=5, stride=2, padding=1
)
assert last_conv_layer(torch.rand(128, 4, 4)).shape == torch.Size([1, 1, 1])


# upconv layer
def upconv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    batch_norm: bool = True,
):
    layers = []
    if stride > 1:
        upsample_layer = nn.Upsample(scale_factor=stride)
        layers.append(upsample_layer)
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
        bias=False,
    )
    layers.append(conv_layer)
    return nn.Sequential(*layers)


# first layer of the generator
print("First layer of the generator")
layer_output = upconv_layer(
    in_channels=100, out_channels=128, kernel_size=5, stride=4, padding=2
)(torch.rand(100, 1, 1))
print(layer_output.shape)  #  == torch.Size([128, 4, 4])
print(layer_output.view(-1, 32, 4, 4).shape)

# second layer of the generator
assert upconv_layer(
    in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=2
)(torch.rand(128, 4, 4)).shape == torch.Size([64, 8, 8])

# third layer of the generator
assert upconv_layer(
    in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=4
)(torch.rand(64, 8, 8)).shape == torch.Size([32, 16, 16])

# fourth layer of the generator
assert upconv_layer(in_channels=32, out_channels=3, kernel_size=1, stride=1, padding=8)(
    torch.rand(32, 16, 16)
).shape == torch.Size([3, 32, 32])

tensor_from_one = torch.tensor([1]).view(1, 1, 1)
tensor_from_zero = torch.tensor([0]).view(1, 1, 1)

# print(tensor_from_one)
# print(tensor_from_one.shape)
# print(tensor_from_one**2)
# print(((tensor_from_one - tensor_from_zero)**2).float().mean())

batch_size = 5
dim = 100
input_noise = torch.rand(batch_size, dim) * 2 - 1
print(input_noise.shape)
print(input_noise.unsqueeze(2).unsqueeze(3).shape)
