# Important Features of the Architecture

![Architecture at training time. See original image in page 3 of [assignment 3](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/assignments/assignment3.pdf)](figures/encoder-decoder.png)

The following building blocks are used to implement the encoder-decoder architecture:

## Encoder (`GRUEncoder`)

- **Components**: A sequence of Embedding layer and GRUCell.
  - **Embedding Layer**: A simple lookup table that stores embeddings of a fixed dictionary (e.g, the tokens themselves) and size (e.g, the number of tokens). For example, the input (dim `batch_size x seq_len`) is a sequence of tokens (e.g., one row is `[7, 1, 9, 14, 9, 14, 7, 28]`) and the output is a tensor of floats (dim `batch_size x seq_len x hidden_size`).
  - **GRU Cell**: Takes each row of the embedding (along the `seq_len` dimension) and feeds it to the GRU. Each step generates an annotation (i.e., the hidden state computed at each step of the input sequence, corresponding to an encoding of each token) and updates the hidden state.
- **Forward Method**: Returns the annotations (e.g., a stack of each word annotation) and the final hidden state.

## `RNNDecoder`

- **Components**: A sequence of Embedding layer, GRUCell, and a Linear layer. *Uses only the encoder hidden states*.
- **Forward Method**: Takes as inputs a tensor of tokenized words, `input`, together with the hidden states of the last step of the encoder, `hidden_init` (dim `(batch_size x hidden_size)`). It computes an embedding over the input and passes it together with `hidden_init` to the GRUCell, which computes a new hidden state. The concatenation of the hidden states generated at each step of the input sequence is then passed through the linear layer to generate a vector of unnormalized log probabilities, later used as an input to the cross-entropy loss function.

## Attention Decoders (`RNNAttentionDecoder`)

- **Components**: A sequence of Embedding layer -> Attention Layer -> GRUCell -> Linear layer. *Uses both the encoder hidden states and the encoder annotations*.
- **Forward Method**:
  - An embedding is computed for each input, but it is not used directly in the GRUCell as in the previous case. Instead, the embedding is fed to the attention layer, together with the encoder annotations, to extract a set of context and attention weights. These are then concatenated along the first dimension and passed as an input to the GRUCell (which has 2x the input dimension as in the previous decoder). The hidden states generated by the GRUCell are then passed to the output Linear layer.
  - **Inputs to the Attention Layer**:
    - **Queries**: Correspond to the current embedding.
    - **Keys and Values**: Given by the annotations (e.g., the encoder hidden states for each step of the input sequence).

### Variations of the Attention Decoder

Within the attention decoder, there are several possible implementations of the attention layer:

1. **AdditiveAttention**:
     - **Components**: A sequence of Linear -> ReLU -> Linear layers (i.e., the attention layer) and a Softmax function.
     - **Forward Method**: Computes the unnormalized attention by concatenating the queries and the keys and passing it through the attention layer. The attention weights are then computed by applying a Softmax to the unnormalized attention. The context is then the product of the values with the attention weights. See the mathematical formula in page 5 of [assignment 3](http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/assignments/assignment3.pdf).

2. **ScaledDotAttention**:
     - This is the usual attention mechanism used in the Transformer model (see, for example, Eq. (1) of the original Transformer paper [here](https://arxiv.org/pdf/1706.03762)).
     - **Components**: 3 linear layers (for Q, K, V, respectively), $W_q$, $W_k$, $W_v$, and a Softmax function.
     - **Forward Method**: Computes the attention weights by taking the dot product of the queries and the keys (weighted by the matrices $W_q$ and $W_k$, respectively), dividing by the square root of the dimension of the keys, and applying a Softmax to the result. The context is then the product of the values with the attention weights.

3. **CausalScaledDotAttention**:
     - **Components**: Identical to the ScaledDotProductAttention.
     - **Forward Method**: Similar to the ScaledDotProductAttention, but with a mask applied to the attention weights to prevent the decoder from attending to future tokens.

## Transformer decoder (`TransformerDecoder`)

- **Components**: An Embedding layer, combined with a self-attention, encoder-attention and attention-mlp layers, used sequentially (such that there are `n_layer` wafers of one self-attention, one encoder-attention and one attention-mlp layer). The output is generated by a Linear layer. *Uses only the encoder annotations*.
    1. **Self-Attention Layer**: Composed by a stack of CausalScaledDotAttention layers, because the decoder should not attend to future tokens.
    2. **Encoder-Attention Layer**: Composed by a stack of ScaledDotAttention layers.
    3. **Attention-MLP Layer**: Composed by a sequence of Linear -> ReLU layers.
- **Forward Method**: The self-attention layer computes new contexts and attention weights for the input sequence using the embedded input (called context) as key, query, and value. The encoder-attention layer then updates the contexts and attention weights using the encoder annotations as key and value, and the sum of the self-attention context and original context as the query. Finally, the attention-mlp layer applies a linear transformation to the sum of the encoder-attention query and its output context (see figure below).

![Example](figures/residual-context-transformer-encoder.png)
