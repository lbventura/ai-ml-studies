import torch
import torch.nn as nn
from rnn_trans_arch.attention import CausalScaledDotAttention, ScaledDotAttention


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.num_layers = num_layers

        self.self_attentions = nn.ModuleList(
            [
                CausalScaledDotAttention(
                    hidden_size=hidden_size,
                )
                for i in range(self.num_layers)
            ]
        )
        self.encoder_attentions = nn.ModuleList(
            [
                ScaledDotAttention(
                    hidden_size=hidden_size,
                )
                for i in range(self.num_layers)
            ]
        )
        self.attention_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                )
                for i in range(self.num_layers)
            ]
        )
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, annotations, hidden_init):
        """Forward pass of the attention-based decoder RNN.

        Arguments:
            inputs: Input token indexes across a batch for all the time step. (batch_size x decoder_seq_len)
            annotations: The encoder hidden states for each step of the input.
                         sequence. (batch_size x seq_len x hidden_size)
            hidden_init: Not used in the transformer decoder
        Returns:
            output: Un-normalized scores for each token in the vocabulary, across a batch for all the decoding time steps. (batch_size x decoder_seq_len x vocab_size)
            attentions: The stacked attention weights applied to the encoder annotations (batch_size x encoder_seq_len x decoder_seq_len)
        """

        batch_size, seq_len = inputs.size()
        embed = self.embedding(inputs)  # batch_size x seq_len x hidden_size

        encoder_attention_weights_list = []
        self_attention_weights_list = []
        contexts = embed
        for i in range(self.num_layers):
            # ------------
            # FILL THIS IN
            # ------------
            # new_contexts, self_attention_weights = ...
            # residual_contexts = ...
            # new_contexts, encoder_attention_weights = ...
            # residual_contexts = ...
            # new_contexts = ...
            # contexts = ...

            # ------------
            # FILL THIS IN
            # ------------
            new_contexts, self_attention_weights = self.self_attentions[i](
                contexts, contexts, contexts
            )
            residual_contexts = contexts + new_contexts
            new_contexts, encoder_attention_weights = self.encoder_attentions[i](
                residual_contexts, annotations, annotations
            )
            residual_contexts = new_contexts + residual_contexts
            new_contexts = self.attention_mlps[i](
                residual_contexts.view(-1, self.hidden_size)
            ).view(batch_size, seq_len, self.hidden_size)
            contexts = new_contexts + residual_contexts

            encoder_attention_weights_list.append(encoder_attention_weights)
            self_attention_weights_list.append(self_attention_weights)

        output = self.out(contexts)
        encoder_attention_weights = torch.stack(encoder_attention_weights_list)
        self_attention_weights = torch.stack(self_attention_weights_list)

        return output, (encoder_attention_weights, self_attention_weights)
