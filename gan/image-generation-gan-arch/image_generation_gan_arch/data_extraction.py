import os
from pathlib import Path
from typing import cast

from six.moves.urllib.request import urlretrieve  # type: ignore
import tarfile

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from image_generation_gan_arch.data_types import InputType


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


def get_emoji_loader(
    emoji_type: InputType,
    image_size: int,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Creates training and test data loaders."""
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),  # Scale
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    data_path = Path(__file__).parent.parent / "data/emojis"

    data_type = cast(str, emoji_type.value)
    train_path = os.path.join(data_path, data_type)
    test_path = os.path.join(data_path, "Test_{}".format(data_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_dloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_dloader, test_dloader
