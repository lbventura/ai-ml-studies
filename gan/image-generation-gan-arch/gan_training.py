import os

import numpy as np
import torch
from image_generation_gan_arch.data_extraction import get_emoji_loader
from image_generation_gan_arch.gan_training_loop import gan_training_loop
from image_generation_gan_arch.data_types import AttrDict
from image_generation_gan_arch.training_utils import print_opts
from pathlib import Path

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)


def train(opts):
    """Loads the data, creates checkpoint and sample directories, and starts the training loop."""

    # Create train and test dataloaders for images from the two domains X and Y
    dataloader_X, test_dataloader_X = get_emoji_loader(emoji_type=opts.X, opts=opts)
    if opts.Y:
        dataloader_Y, test_dataloader_Y = get_emoji_loader(emoji_type=opts.Y, opts=opts)

    # Create checkpoint and sample directories
    parent_path = Path(__file__).parent

    checkpoint_dir_path = parent_path / opts.checkpoint_dir

    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)

    sample_dir_path = parent_path / opts.sample_dir
    if not os.path.exists(sample_dir_path):
        os.makedirs(sample_dir_path)

    # Start training
    if opts.Y is None:
        G, D = gan_training_loop(dataloader_X, test_dataloader_X, opts)
        return G, D
    else:
        #   G_XtoY, G_YtoX, D_X, D_Y = cyclegan_training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, opts)
        #   return G_XtoY, G_YtoX, D_X, D_Y
        pass


if __name__ == "__main__":
    args = AttrDict()
    args_dict = {
        "image_size": 32,
        "g_conv_dim": 32,
        "d_conv_dim": 64,
        "noise_size": 100,
        "num_workers": 0,
        "train_iters": 10000,
        "X": "Windows",  # options: 'Windows' / 'Apple'
        "Y": None,
        "lr": 0.0003,
        "beta1": 0.5,
        "beta2": 0.999,
        "batch_size": 5,
        "checkpoint_dir": "checkpoints_gan",
        "sample_dir": "samples_gan",
        "load": None,
        "log_step": 100,
        "sample_every": 100,
        "checkpoint_every": 100,
    }
    args.update(args_dict)

    print_opts(args)
    G, D = train(args)
