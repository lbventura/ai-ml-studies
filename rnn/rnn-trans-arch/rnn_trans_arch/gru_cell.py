import torch
import torch.nn as nn


class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # ------------
        # FILL THIS IN
        # ------------
        ## Input linear layers
        # The W_i{x} matrices have dimension n_hidden x n_input
        self.Wiz = nn.Linear(
            input_size, hidden_size, bias=False
        )  # torch.randn(hidden_size, input_size).to('cuda')
        self.Wir = nn.Linear(input_size, hidden_size, bias=False)
        self.Win = nn.Linear(input_size, hidden_size, bias=False)

        ## Hidden linear layers
        # The W_h{x} matrices have dimension n_hidden x n_hidden
        self.Whz = nn.Linear(hidden_size, hidden_size)
        self.Whr = nn.Linear(hidden_size, hidden_size)
        self.Whn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        """Forward pass of the GRU computation for one time step.

        Arguments
            x: batch_size x input_size
            h_prev: batch_size x hidden_size

        Returns:
            h_new: batch_size x hidden_size
        """

        # ------------
        # FILL THIS IN
        # ------------
        # works, but worse than the GRUCell implementation

        z_first_term = self.Wiz(x)
        z_second_term = self.Whz(h_prev)
        z = torch.sigmoid(z_first_term + z_second_term)

        r = torch.sigmoid(self.Wir(x) + self.Whr(h_prev))

        g = torch.tanh(self.Win(x) + r * self.Whn(h_prev))
        h_new = (1 - z) * g + z * h_prev
        return h_new
