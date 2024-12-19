import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from image_generation_gan_arch.dcgan import DCGenerator, DCDiscriminator
from image_generation_gan_arch.training_utils import sample_noise
from image_generation_gan_arch.data_extraction import get_emoji_loader

SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device selection
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS device for acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print("Using CUDA device for acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device.")


def load_models(
    saved_folder_path: str, device: torch.device
) -> tuple[DCGenerator, DCDiscriminator]:
    generator = DCGenerator(noise_size=100, conv_dim=32)
    generator.to(DEVICE)

    generator.load_state_dict(
        torch.load(
            os.path.join(saved_folder_path, f"G_{NUMBER_ITERATIONS}.pkl"),
            map_location=DEVICE,
            weights_only=True,
        )
    )

    # Set the model to evaluation mode
    generator.eval()

    discriminator = DCDiscriminator(conv_dim=64)
    discriminator.to(DEVICE)

    discriminator.load_state_dict(
        torch.load(
            os.path.join(saved_folder_path, f"D_{NUMBER_ITERATIONS}.pkl"),
            map_location=DEVICE,
            weights_only=True,
        )
    )

    # Set the model to evaluation mode
    discriminator.eval()
    return generator, discriminator


def test_discriminator(
    batch_size: int,
    device: torch.device,
    generator: DCGenerator,
    discriminator: DCDiscriminator,
) -> np.ndarray:
    noise = sample_noise(batch_size, 100, device=device)  # # 100 x 100 x 1 x 1

    fake_images = generator(noise)
    fake_disc_outputs = discriminator(fake_images)

    zeros_tensor = torch.zeros(batch_size, 1, device=device)
    fake_outputs = torch.cat((fake_disc_outputs.unsqueeze(1), zeros_tensor), dim=1)

    _, test_dataloader_X = get_emoji_loader(
        emoji_type="Windows", image_size=32, batch_size=batch_size
    )

    test_iter = iter(test_dataloader_X)

    test_images, test_labels = next(test_iter)
    test_images, test_labels = (
        test_images.to(device),
        test_labels.long().squeeze().to(device),
    )

    real_disc_outputs = discriminator(test_images)
    ones_tensor = torch.ones(batch_size, 1, device=device)
    real_outputs = torch.cat((real_disc_outputs.unsqueeze(1), ones_tensor), dim=1)

    # Concatenate along the first dimension (dim=0)
    combined_result = torch.cat((fake_outputs, real_outputs), dim=0)  # Shape: [200, 2]

    real_norm_results = F.softmax(combined_result[:, 0], dim=0)
    normalized_results = torch.stack((real_norm_results, combined_result[:, 1]), dim=1)

    # Move tensors to CPU and convert to NumPy arrays
    return normalized_results.cpu().detach().numpy()


def plot_discriminator_results(results: np.ndarray, saved_folder_path: str) -> None:
    x = results[:, 0]
    y = results[:, 1]

    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, x, c=y, cmap="bwr", alpha=0.7)
    plt.xlabel("Normalized Output 1")
    plt.ylabel("Label")
    plt.title("Normalized Results Scatter Plot")
    plt.colorbar(label="Label")
    plt.savefig(os.path.join(saved_folder_path, "normalized_results_plot.jpg"))
    plt.close()

    # Create histograms
    zeros = results[results[:, 1] == 0][:, 0]
    ones = results[results[:, 1] == 1][:, 0]

    # Determine the combined range for both histograms
    min_val = min(zeros.min(), ones.min())
    max_val = max(zeros.max(), ones.max())

    plt.figure(figsize=(10, 6))
    plt.hist(
        zeros, bins=30, alpha=0.5, range=(min_val, max_val), label="Zeros", color="blue"
    )
    plt.hist(
        ones, bins=30, alpha=0.5, range=(min_val, max_val), label="Ones", color="red"
    )
    plt.xlabel("Normalized Output")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normalized Results")
    plt.legend()
    plt.savefig(os.path.join(saved_folder_path, "histogram_normalized_results.jpg"))
    plt.close()


if __name__ == "__main__":
    parent_file = Path(__file__).parent

    MODEL_TYPE = "dcgan"
    EMOJI_TYPE = "Apple"
    LEARNING_RATE = 0.0003
    NUMBER_ITERATIONS = 10000
    BATCH_SIZE = 100

    saved_folder_path = str(
        parent_file
        / f"checkpoints/{MODEL_TYPE}-{EMOJI_TYPE}-lr-{LEARNING_RATE}-train-iter-{NUMBER_ITERATIONS}"
    )

    generator, discriminator = load_models(
        saved_folder_path=saved_folder_path, device=DEVICE
    )

    results = test_discriminator(
        batch_size=BATCH_SIZE,
        device=DEVICE,
        generator=generator,
        discriminator=discriminator,
    )

    plot_discriminator_results(results=results, saved_folder_path=saved_folder_path)
