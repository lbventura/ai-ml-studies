import os
from typing import Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import torch
import torch.nn as nn
from cnn.data_extraction import load_cifar10
from PIL import Image
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss


from cnn.data_preprocessing import get_cat_rgb, get_rgb_cat, process


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
):
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
        outputs.transpose(1, 3).contiguous().view([batch_size * 32 * 32, num_colours])
    )
    loss_lab = labels.transpose(1, 3).contiguous().view([batch_size * 32 * 32])
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
):
    correct = 0.0
    total = 0.0
    losses = []
    num_colours = np.shape(colours)[0]
    for i, (xs, ys) in enumerate(get_batch(test_grey, test_rgb_cat, batch_size)):
        images, labels = get_torch_vars(xs, ys, gpu)
        outputs = cnn(images)

        val_loss = compute_loss(
            criterion, outputs, labels, batch_size=batch_size, num_colours=num_colours
        )
        losses.append(val_loss.data.item())

        _, predicted = torch.max(outputs.data, 1, keepdim=True)
        total += labels.size(0) * 32 * 32
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
):
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
    ]

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


def toimage(img, cmin, cmax):
    return Image.fromarray((img.clip(cmin, cmax) * 255).astype(np.uint8))


def plot_activation(args, cnn):
    # LOAD THE COLOURS CATEGORIES
    colours = np.load(args.colours)[0]
    _ = np.shape(colours)[0]

    (x_train, y_train), (x_test, y_test) = load_cifar10()
    test_rgb, test_grey = process(x_test, y_test, downsize_input=args.downsize_input)
    test_rgb_cat = get_rgb_cat(test_rgb, colours)

    # Take the idnex of the test image
    _id = args.index
    outdir = "outputs/" + args.experiment_name + "/act" + str(id)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    images, labels = get_torch_vars(
        np.expand_dims(test_grey[_id], 0), np.expand_dims(test_rgb_cat[_id], 0)
    )
    cnn.cpu()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1, keepdim=True)
    predcolor = get_cat_rgb(predicted.cpu().numpy()[0, 0, :, :], colours)
    img = predcolor
    toimage(predcolor, cmin=0, cmax=1).save(os.path.join(outdir, f"output_{_id}.png"))

    if not args.downsize_input:
        img = np.tile(np.transpose(test_grey[_id], [1, 2, 0]), [1, 1, 3])
    else:
        img = np.transpose(test_grey[_id], [1, 2, 0])
    toimage(img, cmin=0, cmax=1).save(os.path.join(outdir, f"input_{_id}.png"))

    img = np.transpose(test_rgb[_id], [1, 2, 0])
    toimage(img, cmin=0, cmax=1).save(os.path.join(outdir, f"input_{_id}_gt.png"))

    def add_border(img):
        return np.pad(img, 1, "constant", constant_values=1.0)

    def draw_activations(path, activation, imgwidth=4):
        img = np.vstack(
            [
                np.hstack(
                    [
                        add_border(filter)
                        for filter in activation[
                            i * imgwidth : (i + 1) * imgwidth, :, :
                        ]
                    ]
                )
                for i in range(activation.shape[0] // imgwidth)
            ]
        )
        scipy.misc.imsave(path, img)

    for i, tensor in enumerate([cnn.out1, cnn.out2, cnn.out3, cnn.out4, cnn.out5]):
        draw_activations(
            os.path.join(outdir, f"conv{i + 1}_out_{_id}.png"),
            tensor.data.cpu().numpy()[0],
        )
    print(f"Visualization results are saved to {outdir}")
