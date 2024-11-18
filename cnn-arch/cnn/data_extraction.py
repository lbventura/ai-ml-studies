# adapted from
# https://github.com/fchollet/keras/blob/master/keras/datasets/cifar10.py

import os
import pickle
import tarfile

import numpy as np
from six.moves.urllib.request import urlretrieve  # type: ignore
from pathlib import Path


def get_file(fname: str, origin: str, untar: bool = False) -> str:
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


def load_batch(
    fpath: str, label_key: str = "labels", num_channels: int = 3, image_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Internal utility for parsing CIFAR data.
    # Arguments
        fpath: path the file to parse.
        label_key: key for label data in the retrieve
            dictionary.
    # Returns
        A tuple `(data, labels)`.
    """
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode("utf8")] = v
        d = d_decoded
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], num_channels, image_size, image_size)
    return data, labels


def load_cifar10(
    transpose: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = "cifar-10-batches-py"
    origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(dirname, origin=origin, untar=True)

    num_train_samples = 50000
    n_channels = 3
    image_size = 32

    x_train = np.zeros(
        (num_train_samples, n_channels, image_size, image_size), dtype="uint8"
    )
    y_train = np.zeros((num_train_samples,), dtype="uint8")

    for i in range(1, 6):
        # this works because there are 6 batch files in data
        # numbered from 1 to 5
        fpath = os.path.join(path, "data_batch_" + str(i))
        data, labels = load_batch(fpath, num_channels=n_channels, image_size=image_size)
        x_train[(i - 1) * 10000 : i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000 : i * 10000] = labels

    fpath = os.path.join(path, "test_batch")
    x_test, y_test = load_batch(fpath, num_channels=n_channels, image_size=image_size)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if transpose:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    return (x_train, y_train), (x_test, y_test)


def load_colors() -> str:
    f_path = get_file(
        fname="colours",
        origin="http://www.cs.toronto.edu/~jba/kmeans_colour_a2.tar.gz",
        untar=True,
    )
    return f_path
