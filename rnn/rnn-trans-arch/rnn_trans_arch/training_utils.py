import os
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

import pickle as pkl


def string_to_index_list(
    string: str, char_to_index: dict[str, int], end_token: int
) -> list[int]:
    """Converts a sentence into a list of indexes (for each character)."""
    return [char_to_index[char] for char in string] + [
        end_token
    ]  # Adds the end token to each index list


def translate_sentence(
    sentence: str,
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    cuda: bool,
) -> str:
    """Translates a sentence from English to Pig-Latin, by splitting the sentence into
    words (whitespace-separated), running the encoder-decoder model to translate each
    word independently, and then stitching the words back together with spaces between them.
    """
    return " ".join(
        [translate(word, encoder, decoder, idx_dict, cuda) for word in sentence.split()]
    )


def translate(
    input_string: str,
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    cuda: bool,
) -> str:
    """Translates a given string from English to Pig-Latin."""

    char_to_index: dict[str, int] = idx_dict["char_to_index"]  # type: ignore
    index_to_char: dict[int, str] = idx_dict["index_to_char"]  # type: ignore
    start_token: int = idx_dict["start_token"]  # type: ignore
    end_token: int = idx_dict["end_token"]  # type: ignore

    max_generated_chars = 20
    gen_string = ""

    indexes = string_to_index_list(input_string, char_to_index, end_token)
    indexes = to_var(
        torch.LongTensor(indexes).unsqueeze(0), cuda
    )  # Unsqueeze to make it like BS = 1

    encoder_annotations, encoder_last_hidden = encoder(indexes)

    decoder_hidden = encoder_last_hidden
    decoder_input = to_var(torch.LongTensor([[start_token]]), cuda)  # For BS = 1
    decoder_inputs = decoder_input

    for i in range(max_generated_chars):
        ## slow decoding, recompute everything at each time
        decoder_outputs, attention_weights = decoder(
            decoder_inputs, encoder_annotations, decoder_hidden
        )
        generated_words = F.softmax(decoder_outputs, dim=2).max(2)[1]
        ni = generated_words.cpu().numpy().reshape(-1)  # LongTensor of size 1
        ni = ni[-1]  # latest output token

        decoder_inputs = torch.cat([decoder_input, generated_words], dim=1)

        if ni == end_token:
            break
        else:
            gen_string = "".join(
                [
                    index_to_char[int(item)]
                    for item in generated_words.cpu().numpy().reshape(-1)
                ]
            )

    return gen_string


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


def visualize_attention(
    input_string: str,
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    cuda: bool,
) -> str:
    """Generates a heatmap to show where attention is focused in each decoder step."""
    char_to_index: dict[str, int] = idx_dict["char_to_index"]  # type: ignore
    index_to_char: dict[int, str] = idx_dict["index_to_char"]  # type: ignore
    start_token: int = idx_dict["start_token"]  # type: ignore
    end_token: int = idx_dict["end_token"]  # type: ignore

    max_generated_chars = 20
    gen_string = ""

    indexes = string_to_index_list(input_string, char_to_index, end_token)
    indexes = to_var(
        torch.LongTensor(indexes).unsqueeze(0), cuda
    )  # Unsqueeze to make it like BS = 1

    encoder_annotations, encoder_hidden = encoder(indexes)

    decoder_hidden = encoder_hidden
    decoder_input = to_var(torch.LongTensor([[start_token]]), cuda)  # For BS = 1
    decoder_inputs = decoder_input

    produced_end_token = False

    for i in range(max_generated_chars):
        ## slow decoding, recompute everything at each time
        decoder_outputs, attention_weights = decoder(
            decoder_inputs, encoder_annotations, decoder_hidden
        )
        generated_words = F.softmax(decoder_outputs, dim=2).max(2)[1]
        ni = generated_words.cpu().numpy().reshape(-1)  # LongTensor of size 1
        ni = ni[-1]  # latest output token

        decoder_inputs = torch.cat([decoder_input, generated_words], dim=1)

        if ni == end_token:
            break
        else:
            gen_string = "".join(
                [
                    index_to_char[int(item)]
                    for item in generated_words.cpu().numpy().reshape(-1)
                ]
            )

    if isinstance(attention_weights, tuple):
        ## transformer's attention mweights
        attention_weights, self_attention_weights = attention_weights

    all_attention_weights = attention_weights.data.cpu().numpy()

    for i in range(len(all_attention_weights)):
        attention_weights_matrix = all_attention_weights[i].squeeze()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attention_weights_matrix, cmap="bone")
        fig.colorbar(cax)

        # Set up axes
        ax.set_yticklabels([""] + list(input_string) + ["EOS"], rotation=90)
        ax.set_xticklabels(
            [""] + list(gen_string) + (["EOS"] if produced_end_token else [])
        )

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        # Add title
        plt.xlabel("Attention weights to the source sentence in layer {}".format(i + 1))
        plt.tight_layout()
        plt.grid("off")
        plt.show()
        # plt.savefig(save)

        # plt.close(fig)

    return gen_string


def compute_loss(
    data_dict: dict[tuple[int, int], list[tuple[str, str]]],
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    criterion: _Loss,
    optimizer: Optimizer,
    training_params: TrainingParams,
) -> float:
    """Train/Evaluate the model on a dataset.

    Arguments:
        data_dict: The validation/test word pairs, organized by source and target lengths.
        encoder: An encoder model to produce annotations for each step of the input sequence.
        decoder: A decoder model (with or without attention) to generate output tokens.
        idx_dict: Contains char-to-index and index-to-char mappings, and start & end token indexes.
        criterion: Used to compute the CrossEntropyLoss for each decoder output.
        optimizer: Train the weights if an optimizer is given. None if only evaluate the model.
        training_params: The training parameters.

    Returns:
        mean_loss: The average loss over all batches from data_dict.
    """
    char_to_index: dict[str, int] = idx_dict["char_to_index"]  # type: ignore
    start_token: int = idx_dict["start_token"]  # type: ignore
    end_token: int = idx_dict["end_token"]  # type: ignore

    losses = []
    for key in data_dict:
        input_strings, target_strings = zip(*data_dict[key])
        input_tensors = [
            torch.LongTensor(string_to_index_list(s, char_to_index, end_token))
            for s in input_strings
        ]
        target_tensors = [
            torch.LongTensor(string_to_index_list(s, char_to_index, end_token))
            for s in target_strings
        ]

        num_tensors = len(input_tensors)
        num_batches = int(np.ceil(num_tensors / float(training_params.batch_size)))

        for i in range(num_batches):
            start = i * training_params.batch_size
            end = start + training_params.batch_size

            inputs = to_var(torch.stack(input_tensors[start:end]), training_params.cuda)
            targets = to_var(
                torch.stack(target_tensors[start:end]), training_params.cuda
            )

            # The batch size may be different in each epoch
            BS = inputs.size(0)

            encoder_annotations, encoder_hidden = encoder(inputs)

            start_vector = (
                torch.ones(BS).long().unsqueeze(1) * start_token
            )  # BS x 1 --> 16x1  CHECKED
            decoder_input = to_var(
                start_vector, training_params.cuda
            )  # BS x 1 --> 16x1  CHECKED

            loss = 0.0

            decoder_inputs = torch.cat(
                [decoder_input, targets[:, 0:-1]], dim=1
            )  # Gets decoder inputs by shifting the targets to the right

            decoder_outputs, attention_weights = decoder(
                decoder_inputs, encoder_annotations, encoder_hidden
            )
            decoder_outputs_flatten = decoder_outputs.view(-1, decoder_outputs.size(2))
            targets_flatten = targets.view(-1)
            loss = criterion(decoder_outputs_flatten, targets_flatten)

            losses.append(loss.item())  # type: ignore

            ## training if an optimizer is provided
            if optimizer:
                # Zero gradients
                optimizer.zero_grad()
                # Compute gradients
                loss.backward()  # type: ignore
                # Update the parameters of the encoder and decoder
                optimizer.step()

    mean_loss: float = np.mean(losses)
    return mean_loss


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
        * Prints qualitative translation results each epoch using TEST_SENTENCE
        * Saves an attention map for TEST_WORD_ATTN each epoch

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

    loss_log = open(os.path.join(training_params.checkpoint_dir, "loss_log.txt"), "w")

    best_val_loss = 1e6
    train_losses = []
    val_losses = []

    for epoch in range(training_params.nepochs):
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
        val_loss = compute_loss(
            val_dict,
            encoder,
            decoder,
            idx_dict,
            criterion,
            None,
            training_params=training_params,
        )

        if val_loss < best_val_loss:
            checkpoint(
                encoder,
                decoder,
                idx_dict,
                checkpoint_dir=training_params.checkpoint_dir,
            )

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
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
) -> None:
    """Prints example word pairs, the number of data points, and the vocabulary."""
    print("=" * 80)
    print("Data Stats".center(80))
    print("-" * 80)
    for pair in line_pairs[:5]:
        print(pair)
    print("Num unique word pairs: {}".format(len(line_pairs)))
    print("Vocabulary: {}".format(idx_dict["char_to_index"].keys()))  # type: ignore
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
        "BS={}, nhid={}".format(training_params.batch_size, model_params.hidden_size),
        fontsize=20,
    )
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(training_params.checkpoint_dir, "loss_plot.pdf"))
    plt.close()
