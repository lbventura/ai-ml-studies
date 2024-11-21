import os
from pathlib import Path
import pickle as pkl

from collections import defaultdict

import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from six.moves.urllib.request import urlretrieve  # type: ignore
import tarfile
import torch.nn as nn
from rnn_trans_arch.data_types import AttrDict


def get_file(
    fname: str,
    origin: str,
    untar: bool = False,
) -> str:
    datadir = Path(__file__).parent.parent / "data"
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + ".tar.gz"
    else:
        fpath = os.path.join(datadir, fname)

    print(fpath)
    if not os.path.exists(fpath):
        print("Downloading data from", origin)

        try:
            urlretrieve(origin, fpath)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise e

    if untar:
        if not os.path.exists(untar_fpath):
            print("Extracting file.")
            with tarfile.open(fpath) as archive:
                archive.extractall(datadir)
        return untar_fpath

    return fpath


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


def create_dir_if_not_exists(directory: str) -> None:
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_loss_plot(
    train_losses: list[float], val_losses: list[float], opts: AttrDict
) -> None:
    """Saves a plot of the training and validation loss curves."""
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses)
    plt.plot(range(len(val_losses)), val_losses)
    plt.title("BS={}, nhid={}".format(opts.batch_size, opts.hidden_size), fontsize=20)
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(opts.checkpoint_dir, "loss_plot.pdf"))
    plt.close()


def checkpoint(
    encoder: nn.Module,
    decoder: nn.Module,
    idx_dict: dict[str, dict[str, int] | dict[int, str] | int],
    opts: AttrDict,
) -> None:
    """Saves the current encoder and decoder models, along with idx_dict, which
    contains the char_to_index and index_to_char mappings, and the start_token
    and end_token values.
    """
    with open(os.path.join(opts.checkpoint_dir, "encoder.pt"), "wb") as f:
        torch.save(encoder, f)

    with open(os.path.join(opts.checkpoint_dir, "decoder.pt"), "wb") as f:
        torch.save(decoder, f)

    with open(os.path.join(opts.checkpoint_dir, "idx_dict.pkl"), "wb") as f:
        pkl.dump(idx_dict, f)


def create_dict(
    pairs: list[tuple[str, str]],
) -> dict[tuple[int, int], list[tuple[str, str]]]:
    """Creates a mapping { (source_length, target_length): [list of (source, target) pairs]
    This is used to make batches: each batch consists of two parallel tensors, one containing
    all source indexes and the other containing all corresponding target indexes.
    Within a batch, all the source words are the same length, and all the target words are
    the same length.
    """
    unique_pairs = list(set(pairs))  # Find all unique (source, target) pairs

    d = defaultdict(list)
    for s, t in unique_pairs:
        d[(len(s), len(t))].append((s, t))

    return d


def read_pairs(filename: Path) -> tuple[list[str], list[str]]:
    """Reads lines that consist of two words, separated by a space.

    Returns:
        source_words: A list of the first word in each line of the file.
        target_words: A list of the second word in each line of the file.
    """
    lines = open(filename).read().strip().lower().split("\n")
    source_words, target_words = [], []
    for line in lines:
        line = line.strip()
        if line:
            source, target = line.split()
            source_words.append(source)
            target_words.append(target)
    return source_words, target_words


def all_alpha_or_dash(string: str) -> bool:
    """Helper function to check whether a string is alphabetic, allowing dashes '-'."""
    return all(char.isalpha() or char == "-" for char in string)


def filter_lines(lines: list[str]) -> list[str]:
    """Filters lines to consist of only alphabetic characters or dashes "-"."""
    return [line for line in lines if all_alpha_or_dash(line)]


def load_data() -> (
    tuple[list[tuple[str, str]], int, dict[str, dict[str, int] | dict[int, str] | int]]
):
    """Loads (English, Pig-Latin) word pairs, and creates mappings from characters to indexes."""
    datadir = Path(__file__).parent.parent / "data/pig_latin_data.txt"
    source_lines, target_lines = read_pairs(datadir)

    # Filter lines
    source_lines = filter_lines(source_lines)
    target_lines = filter_lines(target_lines)

    all_characters = set("".join(source_lines)) | set("".join(target_lines))

    # Create a dictionary mapping each character to a unique index
    char_to_index = {
        char: index for (index, char) in enumerate(sorted(list(all_characters)))
    }

    # Add start and end tokens to the dictionary
    start_token = len(char_to_index)
    end_token = len(char_to_index) + 1
    char_to_index["SOS"] = start_token
    char_to_index["EOS"] = end_token

    # Create the inverse mapping, from indexes to characters (used to decode the model's predictions)
    index_to_char = {index: char for (char, index) in char_to_index.items()}

    # Store the final size of the vocabulary
    vocab_size = len(char_to_index)

    line_pairs = list(set(zip(source_lines, target_lines)))  # Python 3

    idx_dict: dict[str, dict[str, int] | dict[int, str] | int] = {
        "char_to_index": char_to_index,
        "index_to_char": index_to_char,
        "start_token": start_token,
        "end_token": end_token,
    }

    return line_pairs, vocab_size, idx_dict
