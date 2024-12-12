import os
from pathlib import Path
from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn_trans_arch.data_types import ModelParams, TrainingParams

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import pickle as pkl

parent_path = Path(__file__).parent.parent

WRITER = SummaryWriter(parent_path / "runs/parameter_updates")


def translate_sentence(
    sentence: str,
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    cuda: bool,
) -> str:
    """Translates a sentence by splitting the sentence into
    words (whitespace-separated), running the encoder-decoder model to translate each
    word independently, and then stitching the words back together with spaces between them.
    """

    return_string = []
    for word in sentence.split():
        generated_word, _ = translate(word, encoder, decoder, idx_dict, cuda)
        return_string.append(generated_word)  # Translates each word

    return " ".join(return_string)


def translate(
    input_string: str,
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    cuda: bool,
) -> tuple[str, torch.Tensor]:
    """Translates a given string using the encoder and decoder."""

    char_to_index = cast(dict[str, int], idx_dict["char_to_index"])
    index_to_char = cast(dict[int, str], idx_dict["index_to_char"])
    start_token = cast(int, idx_dict["start_token"])
    end_token = cast(int, idx_dict["end_token"])

    # Represent the input string as a list of indexes and convert it to a tensor
    indexes = string_to_index_tensor(input_string, char_to_index, end_token)
    indexes = to_var(
        indexes.unsqueeze(0), cuda
    )  # Unsqueeze to make it like batch_size = 1

    generated_tensor, attention_weights = run_encoder_decoder(
        indexes, encoder, decoder, start_token, end_token, cuda
    )

    gen_string = "".join(
        [
            index_to_char[int(item)]
            for item in generated_tensor.cpu().numpy().reshape(-1)
            if index_to_char[int(item)] != "EOS"
        ]
    )

    return gen_string, attention_weights


def run_encoder_decoder(
    indexes: torch.Tensor,
    encoder: nn.Module,
    decoder: nn.Module,
    start_token: int,
    end_token: int,
    cuda: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Encode the input string
    encoder_annotations, encoder_last_hidden = encoder(indexes)

    decoder_input = to_var(
        torch.LongTensor([[start_token]]), cuda
    )  # For batch_size = 1
    # Initialize the decoder input with the start token
    decoder_inputs = decoder_input

    max_generated_chars = 20
    for _ in range(max_generated_chars):
        # slow decoding, recompute everything at each time
        # The alternative would be to store the previous generated words and only run the decoder for the last generated word
        # This, however, assumes that only the last token in the decoder_inputs is different
        decoder_outputs, attention_weights = decoder(
            decoder_inputs, encoder_annotations, encoder_last_hidden
        )
        # This softmax is required because the output layer of the decoder is linear
        generated_tensor = F.softmax(decoder_outputs, dim=2).max(2)[
            1
        ]  # Finds the most likely index for a given token
        # The 0th index contains the probablities of each token in the vocabulary
        ni = generated_tensor.cpu().numpy().reshape(-1)  # LongTensor of size 1
        ni = ni[-1]  # latest output token

        decoder_inputs = torch.cat([decoder_input, generated_tensor], dim=1)

        if ni == end_token:  # If the end token is generated, stop
            break
    return generated_tensor, attention_weights


def string_to_index_tensor(
    string: str, char_to_index: dict[str, int], end_token: int
) -> torch.Tensor:
    """Converts a sentence into a tensor of indices (one for each character).
    The end token is added to each index list.
    """
    return torch.LongTensor([char_to_index[char] for char in string] + [end_token])


def to_var(tensor: torch.Tensor, cuda: bool) -> Variable:
    """Wraps a Tensor in a Variable, optionally placing it on the GPU.

    Arguments:
        tensor: A Tensor object.
        cuda: A boolean flag indicating whether to use the GPU.

    Returns:
        A Variable object, on the GPU if cuda==True.
    """
    if cuda:
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


def training_loop(
    train_dict: dict[tuple[int, int], list[tuple[str, str]]],
    val_dict: dict[tuple[int, int], list[tuple[str, str]]],
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    encoder: nn.Module,
    decoder: nn.Module,
    criterion: _Loss,
    optimizer: Optimizer,
    training_params: TrainingParams,
    model_params: ModelParams,
    test_sentence: str,
) -> None:
    """Runs the main training loop; evaluates the model on the val set every epoch.
        * Prints training and val loss each epoch.
        * Prints qualitative translation results each epoch using test_sentence.

    Arguments:
        train_dict: The training word pairs, organized by source and target lengths.
        val_dict: The validation word pairs, organized by source and target lengths.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model (with or without attention) to generate output tokens.
        criterion: Used to compute the CrossEntropyLoss for each decoder output.
        optimizer: Implements a step rule to update the parameters of the encoder and decoder.
        training_params: The training parameters.
        model_params: The model parameters.
    """
    # Starting decoder parameters
    dec_params = {
        name: param.data.clone() for name, param in decoder.named_parameters()
    }

    with open(
        os.path.join(training_params.checkpoint_dir, "loss_log.txt"), "w"
    ) as loss_log:
        best_val_loss = 1e6
        _tol = 0.05
        train_losses = []
        val_losses = []

        for epoch in range(training_params.nepochs):
            # Decay the learning rate every epoch
            optimizer.param_groups[0]["lr"] *= training_params.lr_decay

            train_loss = compute_loss(
                train_dict,
                encoder,
                decoder,
                idx_dict,
                criterion,
                optimizer,
                training_params=training_params,
            )

            # register parameters
            for name, param in decoder.named_parameters():
                delta = param.data - dec_params[name]
                normalized_norm = torch.norm(delta) / (
                    torch.abs(param.data.mean()) + 1e-7
                )
                WRITER.add_scalar(f"Parameter Updates/{name}", normalized_norm, epoch)
                dec_params[name] = param.data.clone()

            val_loss = compute_loss(
                val_dict,
                encoder,
                decoder,
                idx_dict,
                criterion,
                None,
                training_params=training_params,
            )

            if val_loss < best_val_loss and epoch % 5 == 0:
                best_val_loss = val_loss + _tol
                checkpoint(
                    encoder,
                    decoder,
                    idx_dict,
                    checkpoint_dir=training_params.checkpoint_dir,
                )

            if epoch % 5 == 0:  # translate a test sentence every 5 epochs
                gen_string = translate_sentence(
                    test_sentence, encoder, decoder, idx_dict, cuda=training_params.cuda
                )
                print(
                    "Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f} | Gen: {:20s}".format(
                        epoch, train_loss, val_loss, gen_string
                    )
                )

            loss_log.write("{} {} {}\n".format(epoch, train_loss, val_loss))
            loss_log.flush()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            save_loss_plot(
                train_losses,
                val_losses,
                training_params=training_params,
                model_params=model_params,
            )


def compute_loss(
    data_dict: dict[tuple[int, int], list[tuple[str, str]]],
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    criterion: _Loss,
    optimizer: Optimizer | None,
    training_params: TrainingParams,
) -> float:
    """Train/Evaluate the model on a dataset.

    Arguments:
        data_dict: The validation/test word pairs, organized by source and target lengths. For example, data_dict[(5, 10)] = [('a-day', 'away-ayday')]
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model (with or without attention) to generate output tokens.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        criterion: Used to compute the loss for each decoder output.
        optimizer: Train the weights if an optimizer is given. None if only evaluate the model.
        training_params: The training parameters.

    Returns:
        mean_loss: The average loss over all batches from data_dict.
    """
    char_to_index = cast(dict[str, int], idx_dict["char_to_index"])
    start_token = cast(int, idx_dict["start_token"])
    end_token = cast(int, idx_dict["end_token"])

    losses = []
    for key in data_dict:
        input_strings, target_strings = zip(*data_dict[key])
        input_tensors = [
            string_to_index_tensor(s, char_to_index, end_token) for s in input_strings
        ]  # one tensor per input string
        target_tensors = [
            string_to_index_tensor(s, char_to_index, end_token) for s in target_strings
        ]

        num_batches = int(
            np.ceil(len(input_tensors) / float(training_params.batch_size))
        )  # TODO: It could happen that some batches are very small, is this a problem?

        for i in range(num_batches):
            start = i * training_params.batch_size
            end = start + training_params.batch_size

            inputs = to_var(torch.stack(input_tensors[start:end]), training_params.cuda)
            targets = to_var(
                torch.stack(target_tensors[start:end]), training_params.cuda
            )  # batch_size x seq_len + 1

            # The batch size may be different in each epoch
            batch_size = inputs.size(0)

            # The encoder gets the input strings and produces annotations, as well as the internal hidden states
            # The idea here is similar to a VAE, where the encoder produces a latent representation of the input
            encoder_annotations, encoder_hidden = encoder(inputs)

            start_vector = (
                torch.ones(batch_size).long().unsqueeze(1) * start_token
            )  # batch_size x 1
            decoder_input = to_var(start_vector, training_params.cuda)  # batch_size x 1

            loss = 0.0

            shifted_targets = torch.cat(
                [decoder_input, targets[:, 0:-1]], dim=1
            )  # Gets decoder inputs by shifting the targets to the right
            # See Figure 1 of Assignment 3 here http://www.cs.toronto.edu/~rgrosse/courses/csc421_2019/assignments/assignment3.pdf

            # The decoder gets the target strings, encoder annotations, and the encoder hidden states
            decoder_outputs, _ = decoder(
                shifted_targets, encoder_annotations, encoder_hidden
            )  # Remember that the decoder outputs are the unnormalized scores, not the softmax probabilities
            decoder_outputs_flatten = decoder_outputs.view(
                -1, decoder_outputs.size(2)
            )  # This changes the shape from batch_size x seq_len x vocab_size to batch_size*seq_len x vocab_size
            # This is necessary to compute the loss, which expects a 2D tensor
            targets_flatten = targets.view(
                -1
            )  # This changes the shape from batch_size x seq_len to batch_size*seq_len
            loss = criterion(decoder_outputs_flatten, targets_flatten)

            losses.append(loss.item())  # type: ignore

            # If an optimizer is provided, then the model is in training mode
            if optimizer:
                # Zero gradients
                optimizer.zero_grad()
                # Compute gradients
                loss.backward()  # type: ignore
                # Update the parameters of the encoder and decoder
                optimizer.step()

    mean_loss: float = np.mean(losses)
    return mean_loss


def checkpoint(
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    checkpoint_dir: str,
) -> None:
    """Saves the current encoder and decoder models, along with idx_dict, which
    contains the char_to_index and index_to_char mappings, and the start_token
    and end_token values.
    """
    with open(os.path.join(checkpoint_dir, "encoder.pt"), "wb") as f:
        torch.save(encoder, f)

    with open(os.path.join(checkpoint_dir, "decoder.pt"), "wb") as f:
        torch.save(decoder, f)

    with open(os.path.join(checkpoint_dir, "idx_dict.pkl"), "wb") as f:
        pkl.dump(idx_dict, f)


def print_data_stats(
    line_pairs: list[tuple[str, str]],
    vocab_size: int,
    char_to_index: dict[str, int],
) -> None:
    """Prints example word pairs, the number of data points, and the vocabulary."""
    print("=" * 80)
    print("Data Stats".center(80))
    print("-" * 80)
    for pair in line_pairs[:5]:
        print(pair)
    print("Num unique word pairs: {}".format(len(line_pairs)))
    print("Vocabulary: {}".format(char_to_index.keys()))
    print("Vocab size: {}".format(vocab_size))
    print("=" * 80)


def save_loss_plot(
    train_losses: list[float],
    val_losses: list[float],
    training_params: TrainingParams,
    model_params: ModelParams,
) -> None:
    """Saves a plot of the training and validation loss curves."""
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title(
        "batch_size={}, nhid={}".format(
            training_params.batch_size, model_params.hidden_size
        ),
        fontsize=20,
    )
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(training_params.checkpoint_dir, "loss_plot.pdf"))
    plt.close()


def visualize_attention(
    input_string: str,
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    cuda: bool,
) -> str:
    """Generates a heatmap to show where attention is focused in each decoder step."""
    gen_string, attention_weights = translate(
        input_string=input_string,
        encoder=encoder,
        decoder=decoder,
        idx_dict=idx_dict,
        cuda=cuda,
    )

    if isinstance(attention_weights, tuple):
        ## transformer's attention weights
        attention_weights, _ = attention_weights

    all_attention_weights = attention_weights.data.cpu().numpy()

    for i in range(len(all_attention_weights)):
        attention_weights_matrix = all_attention_weights[i].squeeze()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attention_weights_matrix, cmap="bone")
        fig.colorbar(cax)

        # Set up axes
        ax.set_yticklabels([""] + list(input_string) + ["EOS"], rotation=90)
        ax.set_xticklabels([""] + list(gen_string))

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # Add title
        plt.xlabel("Attention weights to the source sentence in layer {}".format(i + 1))
        plt.tight_layout()
        plt.grid("off")
        plt.show()

    return gen_string


def set_checkpoint_path(
    current_path: Path, training_params: TrainingParams, model_params: ModelParams
) -> None:
    """Set the checkpoint path on the training_params to store model results."""
    input_data_type = training_params.data_source.split(".")[0]
    output_path = current_path / "output" / input_data_type
    model_name = f"h{model_params.hidden_size}-bs{training_params.batch_size}-{model_params.decoder_type}-{model_params.attention_type}"

    training_params.checkpoint_dir = f"{str(output_path)}/{model_name}"
