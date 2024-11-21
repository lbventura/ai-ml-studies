import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):  # type: ignore
    def __init__(self, hidden_size: int):
        super(AdditiveAttention, self).__init__()

        self.hidden_size = hidden_size

        # A two layer fully-connected network
        # hidden_size*2 --> hidden_size, ReLU, hidden_size --> 1
        self.attention_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """The forward pass of the additive attention mechanism.

        Arguments:
            queries: The current decoder hidden state. (batch_size x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x 1 x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The attention_weights must be a softmax weighting over the seq_len annotations.
        """
        batch_size, _, _ = keys.size()
        expanded_queries = queries.unsqueeze(dim=1).expand_as(keys)
        concat_inputs = torch.cat(
            (expanded_queries, keys), dim=2
        )  # the correct dimension to concat along here is the hidden_size, since the attention_network has a shape of hidden_size*2 , hidden_size
        unnormalized_attention = self.attention_network(concat_inputs).view(
            batch_size, -1
        )
        attention_weights = self.softmax(unnormalized_attention).unsqueeze(
            dim=2
        )  # size should be batch_size x seq_len x 1
        context_res = torch.bmm(
            attention_weights.transpose(1, 2), values
        )  # size should be batch_size x 1 x hidden_size
        return context_res, attention_weights


class ScaledDotAttention(nn.Module):  # type: ignore
    def __init__(self, hidden_size: int):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(
            torch.tensor(self.hidden_size, dtype=torch.float)
        )

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """

        batch_size, seq_len, hidden_size = keys.size()

        q = self.Q(queries.view(-1, hidden_size)).view(
            batch_size, -1, hidden_size
        )  # batch_size x (k) x hidden_size
        k = self.K(keys.view(-1, hidden_size)).view(
            batch_size, seq_len, hidden_size
        )  # batch_size x seq_len x hidden_size
        v = self.V(values.view(-1, hidden_size)).view(
            batch_size, seq_len, hidden_size
        )  # batch_size x seq_len x hidden_size
        unnormalized_attention = torch.bmm(
            k, q.transpose(1, 2)
        )  # batch_size x seq_len x k
        attention_weights = self.softmax(
            self.scaling_factor * unnormalized_attention
        )  # batch_size x seq_len x k
        context = torch.bmm(
            attention_weights.transpose(1, 2), v
        )  # batch_size x k x hidden_si

        return context, attention_weights


class CausalScaledDotAttention(nn.Module):  # type: ignore
    def __init__(self, hidden_size: int):
        super(CausalScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size
        self.neg_inf = torch.tensor(-1e7)

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(
            torch.tensor(self.hidden_size, dtype=torch.float)
        )

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """The forward pass of the scaled dot attention mechanism.

        Arguments:
            queries: The current decoder hidden state, 2D or 3D tensor. (batch_size x (k) x hidden_size)
            keys: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)
            values: The encoder hidden states for each step of the input sequence. (batch_size x seq_len x hidden_size)

        Returns:
            context: weighted average of the values (batch_size x k x hidden_size)
            attention_weights: Normalized attention weights for each encoder hidden state. (batch_size x seq_len x 1)

            The output must be a softmax weighting over the seq_len annotations.
        """
        batch_size, seq_len, hidden_size = keys.size()
        q = self.Q(queries.view(-1, hidden_size)).view(
            batch_size, -1, hidden_size
        )  # batch_size x (k) x hidden_size
        k = self.K(keys.view(-1, hidden_size)).view(
            batch_size, seq_len, hidden_size
        )  # batch_size x seq_len x hidden_size
        v = self.V(values.view(-1, hidden_size)).view(
            batch_size, seq_len, hidden_size
        )  # batch_size x seq_len x hidden_size

        # my answer
        # unnormalized_attention = torch.bmm(k, q.transpose(1,2)) #batch_size x seq_len x k

        # solution answer
        unnormalized_attention = self.scaling_factor * torch.bmm(
            k, q.transpose(1, 2)
        )  # batch_size x seq_len x k
        # I think this function is only used in self-attention, where q,k,v have same dimention. k = seq_len

        # TODO: This has some problem with the dimensions, as it should be batch_size x seq_len x k
        # but when it is used in the Decoder, it is size batch_size x seq_len x k
        mask = torch.tril(
            torch.ones(batch_size, seq_len, seq_len, dtype=torch.uint8)
        ).transpose(1, 2)
        unnormalized_attention[mask == 0] = self.neg_inf
        attention_weights = self.softmax(unnormalized_attention)
        context = torch.bmm(attention_weights.transpose(1, 2), v)
        return context, attention_weights
