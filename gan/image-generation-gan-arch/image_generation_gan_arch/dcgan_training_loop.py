import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from image_generation_gan_arch.training_utils import (
    gan_save_samples,
    model_checkpoint,
    print_models,
    sample_noise,
    save_loss_plot,
)
from image_generation_gan_arch.dcgan import DCGenerator, DCDiscriminator
from image_generation_gan_arch.data_types import TrainingParams


def create_model(
    training_params: TrainingParams, device: torch.device
) -> tuple[DCGenerator, DCDiscriminator]:
    """Builds the generators and discriminators for dcgan"""
    G = DCGenerator(
        noise_size=training_params.noise_size, conv_dim=training_params.g_conv_dim
    )
    D = DCDiscriminator(conv_dim=training_params.d_conv_dim)

    print_models(G, None, D, None)

    if device.type == "mps" or device.type == "cuda":
        G.to(device)
        D.to(device)
        print("Models moved to GPU.")
    return G, D


def dcgan_training_loop(
    dataloader: DataLoader,
    test_dataloader: DataLoader,
    training_params: TrainingParams,
    device: torch.device,
) -> dict[str, nn.Module]:
    """Runs the training loop.
    * Saves checkpoint every training_params.checkpoint_every iterations
    * Saves generated samples every training_params.sample_every iterations
    """

    # Create generators and discriminators
    G, D = create_model(training_params=training_params, device=device)

    g_params = G.parameters()  # Get generator parameters
    d_params = D.parameters()  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(
        g_params, training_params.lr, [training_params.beta1, training_params.beta2]
    )
    d_optimizer = optim.Adam(
        d_params,
        training_params.lr * 2.0,
        [training_params.beta1, training_params.beta2],
    )

    train_iter = iter(dataloader)
    test_iter = iter(test_dataloader)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_noise = sample_noise(
        100, training_params.noise_size, device=device
    )  # # 100 x noise_size x 1 x 1

    iter_per_epoch = len(train_iter)
    test_iter_per_epoch = len(test_iter)
    total_train_iters = training_params.train_iters

    train_disc_loss: list[float] = []
    test_disc_loss: list[float] = []
    train_gen_loss: list[float] = []

    try:
        for iteration in range(1, training_params.train_iters + 1):
            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                train_iter = iter(dataloader)
            if iteration % test_iter_per_epoch == 0:
                test_iter = iter(test_dataloader)

            real_images, real_labels = next(train_iter)
            real_images, real_labels = (
                real_images.to(device),
                real_labels.long().squeeze().to(device),
            )

            d_optimizer.zero_grad()

            # 1. Compute the discriminator loss on real images
            D_real_loss = 0.5 * torch.mean((D(real_images) - 1) ** 2)

            # 2. Sample noise
            noise = sample_noise(
                100, training_params.noise_size, device=device
            )  # # 100 x noise_size x 1 x 1

            # 3. Generate fake images from the noise
            fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            D_fake_loss = 0.5 * torch.mean((D(fake_images)) ** 2)

            # 5. Compute the total discriminator loss
            D_total_loss = D_real_loss + D_fake_loss

            # 6. Update discriminator parameters
            D_total_loss.backward()
            d_optimizer.step()

            test_real_images, test_real_labels = next(test_iter)
            test_real_images, test_real_labels = (
                test_real_images.to(device),
                test_real_labels.long().squeeze().to(device),
            )

            ### Compute test discriminator loss
            # 1. Compute the discriminator loss on real images
            test_D_real_loss = 0.5 * torch.mean((D(test_real_images) - 1) ** 2)

            # 2. Sample noise
            noise = sample_noise(
                100, training_params.noise_size, device=device
            )  # # 100 x noise_size x 1 x 1

            # 3. Generate fake images from the noise
            test_fake_images = G(noise)

            # 4. Compute the discriminator loss on the fake images
            test_D_fake_loss = 0.5 * torch.mean((D(test_fake_images)) ** 2)

            # 5. Compute the total discriminator loss
            test_D_total_loss = test_D_real_loss + test_D_fake_loss

            ###########################################
            ###          TRAIN THE GENERATOR        ###
            ###########################################

            g_optimizer.zero_grad()

            # 1. Sample noise
            noise = sample_noise(
                100, training_params.noise_size, device=device
            )  # # 100 x noise_size x 1 x 1

            # 2. Generate fake images from the noise
            fake_images = G(noise)

            # 3. Compute the generator loss
            G_loss = torch.mean((D(fake_images) - 1) ** 2)

            # 4. Update generator parameters
            G_loss.backward()
            g_optimizer.step()

            train_gen_loss.append(G_loss.item())
            train_disc_loss.append(D_total_loss.item())

            test_disc_loss.append(test_D_total_loss.item())

            # Print the log info
            if iteration % training_params.log_step == 0:
                print(
                    "Iteration [{:4d}/{:4d}] | D_real_loss: {:6.4f} | D_fake_loss: {:6.4f} | G_loss: {:6.4f}".format(
                        iteration,
                        total_train_iters,
                        D_real_loss.item(),
                        D_fake_loss.item(),
                        G_loss.item(),
                    )
                )

            # Save the generated samples
            if iteration % training_params.sample_every == 0:
                gan_save_samples(
                    G, fixed_noise, iteration, training_params, device=device
                )

            # Save the model parameters
            if iteration % training_params.checkpoint_every == 0:
                models = {"G": G, "D": D}
                model_checkpoint(
                    iteration=iteration, models=models, training_params=training_params
                )

    except KeyboardInterrupt:
        print("Exiting early from training.")
        return {"generator": G, "discriminator": D}

    save_loss_plot(
        gen_losses=train_gen_loss,
        disc_losses=train_disc_loss,
        training_params=training_params,
    )
    save_loss_plot(
        gen_losses=train_gen_loss,
        disc_losses=test_disc_loss,
        training_params=training_params,
        train=False,
    )

    return {"generator": G, "discriminator": D}
