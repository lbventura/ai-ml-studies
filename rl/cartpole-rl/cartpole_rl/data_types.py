from dataclasses import dataclass


@dataclass
class TrainingParams:
    num_episodes: int = 500
    gamma: float = 0.99
    batch_size: int = 5
    learning_rate: float = 0.01
