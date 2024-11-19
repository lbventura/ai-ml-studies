from typing import Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


from cnn.data_preprocessing import get_cat_rgb
from cnn.data_extraction import IMAGE_SIZE


def get_batch(
    x: np.array, y: np.array, batch_size: int
) -> Generator[tuple[np.array, np.array], None, None]:
    """
    Generator that yields batches of data

    Args:
      x: input values
      y: output values
      batch_size: size of each batch
    Yields:
      batch_x: a batch of inputs of size at most batch_size
      batch_y: a batch of outputs of size at most batch_size
    """
    array_size = np.shape(x)[0]
    assert array_size == np.shape(y)[0]
    for i in range(0, array_size, batch_size):
        batch_x = x[i : i + batch_size, :, :, :]
        batch_y = y[i : i + batch_size, :, :, :]
        yield (batch_x, batch_y)


def get_torch_vars(
    xs: np.array, ys: np.array, gpu: bool = False
) -> tuple[Variable, Variable]:
    """
    Helper function to convert numpy arrays to pytorch tensors.
    If GPU is used, move the tensors to GPU.

    Args:
      xs (float numpy tenosor): greyscale input
      ys (int numpy tenosor): categorical labels
      gpu (bool): whether to move pytorch tensor to GPU
    Returns:
      Variable(xs), Variable(ys)
    """
    xs = torch.from_numpy(xs).float()
    ys = torch.from_numpy(ys).long()
    if gpu:
        xs = xs.cuda()
        ys = ys.cuda()
    return Variable(xs), Variable(ys)


def compute_loss(
    criterion: _Loss,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    num_colours: int,
) -> torch.Tensor:
    """
    Helper function to compute the loss. Since this is a pixelwise
    prediction task we need to reshape the output and ground truth
    tensors into a 2D tensor before passing it in to the loss criteron.

    Args:
      criterion: pytorch loss criterion
      outputs (pytorch tensor): predicted labels from the model
      labels (pytorch tensor): ground truth labels
      batch_size (int): batch size used for training
      num_colours (int): number of colour categories
    Returns:
      pytorch tensor for loss
    """
    loss_out = (
        outputs.transpose(1, 3)
        .contiguous()
        .view([batch_size * IMAGE_SIZE * IMAGE_SIZE, num_colours])
    )
    loss_lab = (
        labels.transpose(1, 3).contiguous().view([batch_size * IMAGE_SIZE * IMAGE_SIZE])
    )
    return criterion(loss_out, loss_lab)


def run_validation_step(
    cnn: nn.Module,
    criterion: _Loss,
    test_grey: np.array,
    test_rgb_cat: np.array,
    batch_size: int,
    gpu: bool,
    colours: np.array,
    plotpath: Optional[str] = None,
    visualize: bool = True,
    downsize_input: bool = False,
) -> tuple[float, torch.Tensor, np.array]:
    correct = 0.0
    total = 0.0
    losses = []
    num_colours = np.shape(colours)[0]
    for _, (xs, ys) in enumerate(get_batch(test_grey, test_rgb_cat, batch_size)):
        images, labels = get_torch_vars(xs, ys, gpu)
        outputs = cnn(images)

        val_loss = compute_loss(
            criterion, outputs, labels, batch_size=batch_size, num_colours=num_colours
        )
        losses.append(val_loss.data.item())

        _, predicted = torch.max(outputs.data, 1, keepdim=True)
        total += labels.size(0) * IMAGE_SIZE * IMAGE_SIZE
        correct += (predicted == labels.data).sum()

    if plotpath:  # only plot if a path is provided
        plot(
            xs,
            ys,
            predicted.cpu().numpy(),
            colours,
            plotpath,
            visualize=visualize,
            compare_bilinear=downsize_input,
        )

    val_loss = np.mean(losses)
    val_acc = 100 * correct / total
    return val_loss, val_acc, predicted


def plot(
    input: np.array,
    gtlabel: np.array,
    output: np.array,
    colours: np.array,
    path: str,
    visualize: bool,
    compare_bilinear: bool = False,
) -> None:
    """
    Generate png plots of input, ground truth, and outputs

    Args:
      input: the greyscale input to the colourization CNN
      gtlabel: the grouth truth categories for each pixel
      output: the predicted categories for each pixel
      colours: numpy array of colour categories and their RGB values
      path: output path
      visualize: display the figures inline or save the figures in path
    """
    grey = np.transpose(input[:10, :, :, :], [0, 2, 3, 1])
    gtcolor = get_cat_rgb(gtlabel[:10, 0, :, :], colours)
    predcolor = get_cat_rgb(output[:10, 0, :, :], colours)

    img_stack = [
        np.hstack(np.tile(grey, [1, 1, 1, 3])),
        np.hstack(gtcolor),
        np.hstack(predcolor),
    ]  # the output is composed of 3 images (from top to bottom): the greyscale input, the ground truth, and the predicted output

    if compare_bilinear:
        downsize_module = nn.Sequential(
            nn.AvgPool2d(2),
            nn.AvgPool2d(2),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Upsample(scale_factor=2, mode="bilinear"),
        )
        gt_input = np.transpose(
            gtcolor,
            [
                0,
                3,
                1,
                2,
            ],
        )
        color_bilinear = downsize_module.forward(torch.from_numpy(gt_input).float())
        color_bilinear = np.transpose(color_bilinear.data.numpy(), [0, 2, 3, 1])
        img_stack = [
            np.hstack(np.transpose(input[:10, :, :, :], [0, 2, 3, 1])),
            np.hstack(gtcolor),
            np.hstack(predcolor),
            np.hstack(color_bilinear),
        ]
    img = np.vstack(img_stack)

    plt.grid("off")
    plt.imshow(img, vmin=0.0, vmax=1.0)
    if visualize:
        plt.show()
    else:
        plt.savefig(path)
