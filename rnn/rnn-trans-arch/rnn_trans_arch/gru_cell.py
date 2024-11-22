import torch
import torch.nn as nn


class MyGRUCell(nn.Module):  # type: ignore
    """
    A custom implementation of a Gated Recurrent Unit (GRU) cell.
    This class defines the operations for a single GRU cell, which can be used
    as a building block for constructing recurrent neural networks. The GRU cell
    is designed to handle sequential data and maintain hidden state information
    across time steps.
    Note: This is not used neither in the GRU nor in the RNNDecoder, but it is useful for understanding how it works.
    Attributes:
        input_size (int): The number of input features.
        hidden_size (int): The number of features in the hidden state.
    Methods:
        forward(x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
            Performs the forward pass of the GRU cell for one time step.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ## Input linear layers
        # The W_i{x} matrices have dimension n_hidden x n_input
        self.Wiz = nn.Linear(input_size, hidden_size, bias=False)
        self.Wir = nn.Linear(input_size, hidden_size, bias=False)
        self.Win = nn.Linear(input_size, hidden_size, bias=False)

        ## Hidden linear layers
        # The W_h{x} matrices have dimension n_hidden x n_hidden
        self.Whz = nn.Linear(hidden_size, hidden_size)
        self.Whr = nn.Linear(hidden_size, hidden_size)
        self.Whn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """
        z_first_term = self.Wiz(x)
        z_second_term = self.Whz(h_prev)
        z = torch.sigmoid(z_first_term + z_second_term)

        r = torch.sigmoid(self.Wir(x) + self.Whr(h_prev))

        g = torch.tanh(self.Win(x) + r * self.Whn(h_prev))
        h_new = (1 - z) * g + z * h_prev
        return h_new
