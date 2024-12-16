import torch.nn as nn

import numpy as np
import torch
from image_generation_gan_arch.data_extraction import get_emoji_loader
from image_generation_gan_arch.cyclegan_training_loop import cyclegan_training_loop
from image_generation_gan_arch.dcgan_training_loop import dcgan_training_loop
from image_generation_gan_arch.data_types import ModelType, TrainingParams
from image_generation_gan_arch.training_utils import create_directories, print_opts
from pathlib import Path
import time

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS device for acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print("Using CUDA device for acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device.")


def train(
    training_params: TrainingParams, device: torch.device = DEVICE
) -> dict[str, nn.Module]:
    """Loads the data, creates checkpoint and sample directories, and starts the training loop."""

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(
        emoji_type=training_params.X, training_params=training_params
    )
    if training_params.model_type == ModelType.cyclegan:
        dataloader_Y, test_dataloader_Y = get_emoji_loader(
            emoji_type=training_params.Y, training_params=training_params
        )

    # Create checkpoint and sample directories
    parent_path = Path(__file__).parent

    create_directories(parent_path=parent_path, training_params=training_params)

    # Start training
    if training_params.model_type == ModelType.dcgan:
        return dcgan_training_loop(
            dataloader_X,
            test_dataloader_X,
            training_params=training_params,
            device=device,
        )

    else:
        return cyclegan_training_loop(
            dataloader_X,
            dataloader_Y,
            test_dataloader_X,
            test_dataloader_Y,
            training_params=training_params,
            device=device,
        )


if __name__ == "__main__":
    start_time = time.time()
    training_params = TrainingParams(model_type=ModelType.cyclegan, lambda_cycle=0.03)
    print_opts(training_params=training_params)
    train(training_params=training_params)
    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")
