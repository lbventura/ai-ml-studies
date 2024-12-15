from dataclasses import dataclass


@dataclass
class TrainingParams:
    image_size: int = 32
    g_conv_dim: int = 32
    d_conv_dim: int = 64
    noise_size: int = 100
    num_workers: int = 0
    train_iters: int = 1000
    X: str = "Windows"
    Y: str | None = None
    lr: float = 0.0003
    beta1: float = 0.5
    beta2: float = 0.999
    batch_size: int = 5
    checkpoint_dir: str = "checkpoints_gan"
    sample_dir: str = "samples_gan"
    load: str | None = None
    log_step: int = 100
    sample_every: int = 100
    checkpoint_every: int = 100
    init_zero_weights: bool = False
