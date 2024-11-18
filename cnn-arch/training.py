import os
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from cnn.cnnet import CNN
from cnn.data_extraction import load_cifar10, load_colors
from cnn.data_preprocessing import (
    get_rgb_cat,
    process,
)
from cnn.training_utils import (
    get_batch,
    get_torch_vars,
    compute_loss,
    plot,
    run_validation_step,
)

from cnn.unet import UNet

from pathlib import Path
from dataclasses import dataclass


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


def train(args: ModelParams):
    torch.set_num_threads(5)
    # Numpy random seed
    npr.seed(args.seed)

    # Save directory
    parent_path = Path(__file__).parent
    save_dir = parent_path / f"outputs/{args.experiment_name}"
    colors_dir = load_colors() + "/colour_kmeans24_cat7.npy"

    # LOAD THE COLOURS CATEGORIES
    colours = np.load(colors_dir, allow_pickle=True, encoding="bytes")[0]
    num_colours = np.shape(colours)[0]

    # INPUT CHANNEL
    num_in_channels = 1 if not args.downsize_input else 3

    print(f"The number of num_in_channels is {num_in_channels}")
    print(f"The number of num_colours is {num_colours}")

    # LOAD THE MODEL
    if args.model == "CNN":
        cnn = CNN(args.kernel, args.num_filters, num_colours, num_in_channels)
    elif args.model == "UNet":
        cnn = UNet(args.kernel, args.num_filters, num_colours, num_in_channels)

    # LOSS FUNCTION
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=args.learn_rate)

    # DATA
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Transforming data...")
    train_rgb, train_grey = process(
        x_train, y_train, downsize_input=args.downsize_input
    )
    train_rgb_cat = get_rgb_cat(train_rgb, colours)
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input)
    test_rgb_cat = get_rgb_cat(test_rgb, colours)

    # Create the outputs folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Beginning training ...")
    if args.gpu:
        cnn.cuda()
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(args.epochs):
        # Train the Model
        cnn.train()  # Change model to 'train' mode
        losses = []
        for _, (xs, ys) in enumerate(
            get_batch(train_grey, train_rgb_cat, args.batch_size)
        ):
            images, labels = get_torch_vars(xs, ys, args.gpu)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)

            loss = compute_loss(
                criterion,
                outputs,
                labels,
                batch_size=args.batch_size,
                num_colours=num_colours,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

        # plot training images
        if args.plot:
            plt_path = str(save_dir / f"train_{args.model}_{epoch}.png")
            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            plot(
                xs,
                ys,
                predicted.cpu().numpy(),
                colours,
                plt_path,
                args.visualize,
                args.downsize_input,
            )

        # plot training images
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print(
            f"Epoch [{epoch + 1}/{args.epochs}], Loss: {round(avg_loss,4)}, Time (s): {round(time_elapsed,4)}"
        )

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        val_loss, val_acc, predicted = run_validation_step(
            cnn,
            criterion,
            test_grey,
            test_rgb_cat,
            args.batch_size,
            args.gpu,
            colours,
            # save_dir+'/test_%d.png' % epoch,
            visualize=args.visualize,
            downsize_input=args.downsize_input,
        )

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print(
            "Epoch [%d/%d], Val Loss: %.4f, Val Acc: %.1f%%, Time(s): %d"
            % (epoch + 1, args.epochs, val_loss, val_acc, time_elapsed)
        )

    # Plot training curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir / "training_curve.png")

    if args.checkpoint:
        print("Saving model...")
        torch.save(cnn.state_dict(), args.checkpoint)

    return cnn, predicted


if __name__ == "__main__":
    args = ModelParams()
    cnn, predicted = train(args)
