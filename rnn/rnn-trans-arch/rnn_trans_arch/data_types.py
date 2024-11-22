from dataclasses import dataclass
from enum import StrEnum


@dataclass
class TrainingParams:
    cuda: bool = False
    nepochs: int = 75
    checkpoint_dir: str = "checkpoints"
    learning_rate: float = 0.005
    lr_decay: float = 0.99
    batch_size: int = 64


class DecoderType(StrEnum):
    rnn = "rnn"
    rnn_attention = "rnn_attention"
    transformer = "transformer"


class AttentionType(StrEnum):
    additive = "additive"
    scaled_dot = "scaled_dot"
    causal_scaled_dot = "causal_scaled_dot"


@dataclass
class ModelParams:
    hidden_size: int = 20
    decoder_type: DecoderType = DecoderType.rnn_attention
    attention_type: AttentionType = AttentionType.additive
    num_transformer_layers: int | None = None

    def __post_init__(self) -> None:
        if self.decoder_type == DecoderType.transformer:
            assert (
                self.num_transformer_layers is not None
            ), "num_transformer_layers must be set for Transformer models, the default value is 3."
