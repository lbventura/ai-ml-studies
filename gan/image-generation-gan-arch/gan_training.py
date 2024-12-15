import os
from typing import Any

import numpy as np
import torch
from image_generation_gan_arch.data_extraction import get_emoji_loader
from image_generation_gan_arch.dcgan_training_loop import dcgan_training_loop
from image_generation_gan_arch.data_types import TrainingParams
from image_generation_gan_arch.training_utils import print_opts
from pathlib import Path

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


def train(training_params: TrainingParams, device: torch.device = DEVICE) -> Any:
    """Loads the data, creates checkpoint and sample directories, and starts the training loop."""

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(
        emoji_type=training_params.X, training_params=training_params
    )
    if training_params.Y:
        dataloader_Y, test_dataloader_Y = get_emoji_loader(
            emoji_type=training_params.Y, training_params=training_params
        )

    # Create checkpoint and sample directories
    parent_path = Path(__file__).parent

    checkpoint_dir_path = parent_path / training_params.checkpoint_dir

    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)

    sample_dir_path = parent_path / training_params.sample_dir
    if not os.path.exists(sample_dir_path):
        os.makedirs(sample_dir_path)

    # Start training
    if training_params.Y is None:
        G, D = dcgan_training_loop(
            dataloader_X,
            test_dataloader_X,
            training_params=training_params,
            device=device,
        )
        return G, D
    else:
        #   G_XtoY, G_YtoX, D_X, D_Y = cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, training_params)
        #   return G_XtoY, G_YtoX, D_X, D_Y
        pass


if __name__ == "__main__":
    training_params = TrainingParams()
    print_opts(training_params=training_params)
    G, D = train(training_params=training_params)
