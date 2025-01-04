from torch import nn
import torch.nn.functional as F
import torch


class PolicyNet(nn.Module):  # type: ignore
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(PolicyNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.relu(self.fc1(x))
        output = F.relu(self.fc2(output))
        output = torch.sigmoid(self.output(output))

        return output
