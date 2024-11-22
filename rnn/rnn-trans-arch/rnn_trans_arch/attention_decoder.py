import torch
import torch.nn as nn

from rnn_trans_arch.attention import (
    AdditiveAttention,
    CausalScaledDotAttention,
    ScaledDotAttention,
)


class RNNAttentionDecoder(nn.Module):  # type: ignore
    def __init__(
        self, vocab_size: int, hidden_size: int, attention_type: str = "scaled_dot"
    ):
        super(RNNAttentionDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        self.rnn = nn.GRUCell(input_size=hidden_size * 2, hidden_size=hidden_size)
        if attention_type == "additive":
            self.attention = AdditiveAttention(hidden_size=hidden_size)
        elif attention_type == "scaled_dot":
            self.attention = ScaledDotAttention(hidden_size=hidden_size)
        elif attention_type == "causal_scaled_dot":
            self.attention = CausalScaledDotAttention(hidden_size=hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, inputs: torch.Tensor, annotations: torch.Tensor, hidden_init: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: The final hidden states from the encoder, across a batch. (batch_size x hidden_size)

        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """

        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        hiddens = []
        attentions = []
        h_prev = hidden_init
        for i in range(seq_len):
            embed_current = embed[:, i, :]
            context_res, attention_weights = self.attention(
                queries=embed_current, keys=annotations, values=annotations
            )
            embed_and_context = torch.cat(
                (embed_current, context_res.squeeze(dim=1)), dim=1
            )  # so that both inputs have dim batch_size * hidden_size
            h_prev = self.rnn(embed_and_context, h_prev)

            hiddens.append(h_prev)
            attentions.append(attention_weights)

        hiddens = torch.stack(hiddens, dim=1)  # batch_size x seq_len x hidden_size
        attentions = torch.cat(attentions, dim=2)  # batch_size x seq_len x seq_len

        output = self.out(hiddens)  # batch_size x seq_len x vocab_size
        return output, attentions
