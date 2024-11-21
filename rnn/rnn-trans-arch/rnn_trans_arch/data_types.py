from dataclasses import dataclass


@dataclass
class AttrDict:
    cuda: bool = False
    nepochs: int = 1
    checkpoint_dir: str = "checkpoints"
    learning_rate: float = 0.005
    lr_decay: float = 0.99
    batch_size: int = 64
    hidden_size: int = 20
    decoder_type: str = "rnn_attention"  # options: rnn / rnn_attention / transformer
    attention_type: str = "additive"  # options: additive / scaled_dot
    num_transformer_layers: int = 3
