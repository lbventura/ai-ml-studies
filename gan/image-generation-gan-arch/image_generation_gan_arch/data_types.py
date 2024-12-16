from dataclasses import dataclass
from enum import StrEnum


class ModelType(StrEnum):
    dcgan = "dcgan"
    cyclegan = "cyclegan"


@dataclass
class TrainingParams:
    image_size: int = 32
    g_conv_dim: int = 32
    d_conv_dim: int = 64
    noise_size: int = 100
    num_workers: int = 0
    train_iters: int = 1000
    model_type: ModelType = ModelType.dcgan
    X: str = "Windows"
    lr: float = 0.0003
    beta1: float = 0.5
    beta2: float = 0.999
    batch_size: int = 5
    load: str | None = None
    log_step: int = 100
    sample_every: int = 100
    checkpoint_every: int = 100
    init_zero_weights: bool = False
    lambda_cycle: float | None = None

    def __post_init__(self) -> None:
        self.checkpoint_dir = f"checkpoints/{self.model_type}-{self.X}-lr-{self.lr}-train-iter-{self.train_iters}"
        self.sample_dir = f"samples/{self.model_type}-{self.X}-lr-{self.lr}-train-iter-{self.train_iters}"

        if self.model_type == ModelType.cyclegan and self.X == "Windows":
            self.Y = "Apple"
        elif self.model_type == ModelType.cyclegan and self.X == "Apple":
            self.Y = "Windows"

        if self.model_type == ModelType.cyclegan and self.lambda_cycle is None:
            raise ValueError(
                "lambda_cycle must be set for CycleGAN models. A good default value is 0.03"
            )
