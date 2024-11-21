import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
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

    def forward(self, queries, keys, values):
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

        # ------------
        # FILL THIS IN
        # ------------
        # batch_size = ... # this is the first dimension of the query
        # expanded_queries = ... # because the keys have dimension batch_size * seq_len * hidden_size, we might have to take the queries and expand_as the keys: queries.expand_as(keys)
        # concat_inputs = ... # such that we are then able to concatenate the two tensors along the hidden dimension (which is dim 1 - the columns of the query)
        # unnormalized_attention = ... # apply the attention network to concat_inputs
        # attention_weights = ... # apply the softmax to unnormalized_attention - this should have dimension batch_size * seq_len * 1
        # context = ... # compute the dot product between the attention_weights and the values (which have dimension batch_size * seq_len * hidden_size) - this should have dimension batch_size * 1 * hidden_size
        # the dot product is probably along the seq_len dimension
        # return context, attention_weights
        batch_size, seq_len, hidden_size = keys.size()
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


class ScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotAttention, self).__init__()

        self.hidden_size = hidden_size

        self.Q = nn.Linear(hidden_size, hidden_size)
        self.K = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=1)
        self.scaling_factor = torch.rsqrt(
            torch.tensor(self.hidden_size, dtype=torch.float)
        )

    def forward(self, queries, keys, values):
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

        # ------------
        # FILL THIS IN
        # ------------
        # batch_size = ...
        # q = ...
        # k = ...
        # v = ...
        # unnormalized_attention = ...
        # attention_weights = ...
        # context = ...
        # return context, attention_weights
        batch_size, seq_len, hidden_size = keys.size()
        # My answer -- works but results are worse than the linear attention
        # q = self.Q
        # k = self.K
        # v = self.V

        # k_term = k(keys.view(keys.size(2), -1).transpose(0,1)).view(batch_size, seq_len, hidden_size) # keys has shape (batch_size x seq_len x hidden_size)
        # # since k has shape (hidden_size, hidden_size), we first have to transform keys to (batch_size*seq_len x hidden_size) and then transpose
        # # (hidden_size x batch_size*seq_len)

        # if len(queries.shape) == 3:
        #   input_queries = queries.view(queries.size(2), -1)
        #   k = queries.shape[2]
        # else:
        #   input_queries = queries
        #   k = 1
        # q_term = q(input_queries).view(batch_size, k, hidden_size)

        # unnormalized_attention = torch.bmm(q_term, k_term.transpose(1,2))
        # attention_weights = self.softmax(unnormalized_attention * self.scaling_factor)

        # v_term = v((values.view(values.size(2), -1)).transpose(0,1)).view(batch_size, seq_len, hidden_size)
        # context = torch.bmm(attention_weights, v_term)

        # My answer modified to understand where the source of the worsening performance is
        # q = self.Q
        # k = self.K
        # v = self.V

        # k_term = self.K(keys.view(-1,hidden_size)).view(batch_size, seq_len, hidden_size) # keys has shape (batch_size x seq_len x hidden_size)
        # # since k has shape (hidden_size, hidden_size), we first have to transform keys to (batch_size*seq_len x hidden_size) and then transpose
        # # (hidden_size x batch_size*seq_len)

        # q_term = self.Q(queries.view(-1, hidden_size)).view(batch_size, -1, hidden_size)

        # unnormalized_attention = torch.bmm(q_term, k_term.transpose(1,2)) # Maybe this is the source of the error??
        # attention_weights = self.softmax(unnormalized_attention * self.scaling_factor)

        # v_term = self.V(values.view(-1, hidden_size)).view(batch_size, seq_len, hidden_size)
        # context = torch.bmm(attention_weights, v_term)

        # Solutions answer
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


class CausalScaledDotAttention(nn.Module):
    def __init__(self, hidden_size):
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

    def forward(self, queries, keys, values):
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

        # ------------
        # FILL THIS IN
        # ------------
        # batch_size = ...
        # q = ...
        # k = ...
        # v = ...
        # unnormalized_attention = ...
        # mask = ...
        # attention_weights = ...
        # context = ...
        # return context, attention_weights

        # Solutions answer
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

        # # Set elements above the main diagonal to the desired value
        # causal_unnormalized_attention = unnormalized_attention * mask + self.neg_inf * (1 - mask)

        # attention_weights = self.softmax(self.scaling_factor * causal_unnormalized_attention) #batch_size x seq_len x k
        # context = torch.bmm(attention_weights.transpose(1,2), v) #batch_size x k x hidden_si

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
