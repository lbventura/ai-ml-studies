from dataclasses import dataclass
from enum import Enum

import numpy as np


class Categories(Enum):
    DOGS = 5
    FROGS = 6
    HORSES = 7


@dataclass
class TrainingParams:
    gpu: bool = False
    checkpoint: str = ""
    learn_rate: float = 0.001
    batch_size: int = 25
    epochs: int = 5
    seed: int = 0
    plot: bool = True
    visualize: bool = False
    downsize_input: bool = False
    input_category: Categories = Categories.HORSES


@dataclass
class ModelParams:
    model: str = "CNN"
    kernel: int = 3
    num_filters: int = 32

    def __post_init__(self):
        self.experiment_name = f"colourization_{self.model.lower()}_kernel_{self.kernel}_filters_{self.num_filters}"


@dataclass
class ModelData:
    train_grey: np.array
    train_rgb_cat: np.array
    test_grey: np.array
    test_rgb_cat: np.array
