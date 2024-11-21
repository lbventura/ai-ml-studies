import os
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
from cnn.cnnet import CNN
from cnn.data_extraction import load_cifar10, load_colors
from cnn.data_preprocessing import get_rgb_cat, process
from cnn.training_utils import (
    get_batch,
    compute_loss,
    get_torch_vars,
    plot,
    run_validation_step,
)

from cnn.unet import UNet

from pathlib import Path
from cnn.data_types import ModelParams, TrainingParams, ModelData


def _prepare_data(training_args: TrainingParams) -> tuple[np.array, ModelData]:
    colors_dir = load_colors() + "/colour_kmeans24_cat7.npy"
    # LOAD THE COLOURS CATEGORIES
    colours = np.load(colors_dir, allow_pickle=True, encoding="bytes")[0]
    print(f"The number of num_colours is {np.shape(colours)[0]}")

    # LOAD DATA
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_cifar10()

    print("Transforming data...")
    train_rgb, train_grey = process(
        x_train,
        y_train,
        downsize_input=training_args.downsize_input,
        category=training_args.input_category,
    )
    train_rgb_cat = get_rgb_cat(train_rgb, colours)

    test_rgb, test_grey = process(
        x_test,
        y_test,
        downsize_input=training_args.downsize_input,
        category=training_args.input_category,
    )
    test_rgb_cat = get_rgb_cat(test_rgb, colours)

    return (
        colours,
        ModelData(
            train_grey=train_grey,
            train_rgb_cat=train_rgb_cat,
            test_grey=test_grey,
            test_rgb_cat=test_rgb_cat,
        ),
    )


def train(
    training_args: TrainingParams, model_args: ModelParams
) -> tuple[nn.Module, np.array]:
    torch.set_num_threads(5)
    # Numpy random seed
    npr.seed(training_args.seed)

    # Save directory
    save_dir = str(Path(__file__).parent) + f"/outputs/{model_args.experiment_name}"

    # Create the outputs folder if not created already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # INPUT CHANNEL
    num_in_channels = 1 if not training_args.downsize_input else 3

    print(f"The number of num_in_channels is {num_in_channels}")

    # PREPARE DATA
    colours, model_data = _prepare_data(training_args=training_args)
    num_colours = np.shape(colours)[0]

    # SETUP THE MODEL
    if model_args.model == "CNN":
        cnn = CNN(
            model_args.kernel, model_args.num_filters, num_colours, num_in_channels
        )
    elif model_args.model == "UNet":
        cnn = UNet(
            model_args.kernel, model_args.num_filters, num_colours, num_in_channels
        )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=training_args.learn_rate)

    print(f"Beginning training on category: {training_args.input_category}")
    if training_args.gpu:
        cnn.cuda()
    start = time.time()

    train_losses = []
    valid_losses = []
    valid_accs = []
    for epoch in range(training_args.epochs):
        # Train the Model
        cnn.train()  # Change model to 'train' mode
        losses = []
        for _, (xs, ys) in enumerate(
            get_batch(
                model_data.train_grey,
                model_data.train_rgb_cat,
                training_args.batch_size,
            )
        ):
            optimizer.zero_grad()

            images, labels = get_torch_vars(xs, ys, training_args.gpu)
            # Forward + Backward + Optimize
            outputs = cnn(images)

            loss = compute_loss(
                criterion,
                outputs,
                labels,
                batch_size=training_args.batch_size,
                num_colours=num_colours,
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.data.item())

        # plot training images
        if training_args.plot:
            plt_path = save_dir + f"/train_{model_args.model}_{epoch}.png"
            _, predicted = torch.max(outputs.data, 1, keepdim=True)
            plot(
                xs,
                ys,
                predicted.cpu().numpy(),
                colours,
                plt_path,
                training_args.visualize,
                training_args.downsize_input,
            )

        # Compute loss and accuracy
        avg_loss = np.mean(losses)
        train_losses.append(avg_loss)
        time_elapsed = time.time() - start
        print(
            f"Epoch [{epoch + 1}/{training_args.epochs}], Loss: {round(avg_loss,4)}, Time (s): {round(time_elapsed,4)}"
        )

        # Evaluate the model
        cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        val_loss, val_acc, predicted = run_validation_step(
            cnn,
            criterion,
            model_data.test_grey,
            model_data.test_rgb_cat,
            training_args.batch_size,
            training_args.gpu,
            colours,
            # save_dir+'/test_%d.png' % epoch,
            visualize=training_args.visualize,
            downsize_input=training_args.downsize_input,
        )

        time_elapsed = time.time() - start
        valid_losses.append(val_loss)
        valid_accs.append(val_acc)
        print(
            f"Epoch {epoch + 1}/{training_args.epochs}, Val Loss: {round(val_loss, 4)}, Val Acc: {round(val_acc.item(), 1)}%, Time(s): {round(time_elapsed,0)}"
        )

    # Plot training curve
    plt.figure()
    plt.plot(train_losses, "ro-", label="Train")
    plt.plot(valid_losses, "go-", label="Validation")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.savefig(save_dir + "/training_curve.png")

    if training_args.checkpoint:
        print("Saving model...")
        torch.save(cnn.state_dict(), training_args.checkpoint)

    return cnn, predicted


if __name__ == "__main__":
    training_args = TrainingParams(epochs=25)
    model_args = ModelParams(model="UNet")
    cnn, predicted = train(training_args=training_args, model_args=model_args)
