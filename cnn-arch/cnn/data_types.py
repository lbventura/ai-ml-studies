from dataclasses import dataclass
from enum import Enum


class Categories(Enum):
    DOGS = 5
    FROGS = 6
    HORSES = 7


@dataclass
class ModelParams:
    gpu: bool = False
    valid: bool = False
    checkpoint: str = ""
    model: str = "CNN"
    kernel: int = 3
    num_filters: int = 32
    learn_rate: float = 0.001
    batch_size: int = 25
    epochs: int = 5
    seed: int = 0
    plot: bool = True
    experiment_name: str = "colourization_cnn"
    visualize: bool = False
    downsize_input: bool = False
    input_category: Categories = Categories.HORSES
    index: int = 0
