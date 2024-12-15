import torch
import torch.optim as optim
import torch.nn as nn

from image_generation_gan_arch.training_utils import (
    create_model,
    cyclegan_checkpoint,
    cyclegan_save_samples,
)
from torch.utils.data import DataLoader
from image_generation_gan_arch.dcgan import DCDiscriminator
from image_generation_gan_arch.cyclegan import CycleGenerator
from image_generation_gan_arch.data_types import TrainingParams


def cyclegan_training_loop(
    dataloader_X: DataLoader,
    dataloader_Y: DataLoader,
    test_dataloader_X: DataLoader,
    test_dataloader_Y: DataLoader,
    training_params: TrainingParams,
    device: torch.device,
) -> dict[str, nn.Module]:
    """Runs the training loop.
    * Saves checkpoint every training_params.checkpoint_every iterations
    * Saves generated samples every training_params.sample_every iterations
    """

    # Create generators and discriminators
    G_XtoY, G_YtoX, D_X, D_Y = create_model(
        training_params, CycleGenerator, DCDiscriminator, device
    )

    g_params = list(G_XtoY.parameters()) + list(
        G_YtoX.parameters()
    )  # Get generator parameters
    d_params = list(D_X.parameters()) + list(
        D_Y.parameters()
    )  # Get discriminator parameters

    # Create optimizers for the generators and discriminators
    g_optimizer = optim.Adam(
        g_params, training_params.lr, [training_params.beta1, training_params.beta2]
    )
    d_optimizer = optim.Adam(
        d_params, training_params.lr, [training_params.beta1, training_params.beta2]
    )

    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)

    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    # Get some fixed data from domains X and Y for sampling. These are images that are held
    # constant throughout training, that allow us to inspect the model's performance.
    fixed_X = (next(test_iter_X)[0]).to(device)
    fixed_Y = (next(test_iter_Y)[0]).to(device)

    iter_per_epoch = min(len(iter_X), len(iter_Y))

    try:
        for iteration in range(1, training_params.train_iters + 1):
            # Reset data_iter for each epoch
            if iteration % iter_per_epoch == 0:
                iter_X = iter(dataloader_X)
                iter_Y = iter(dataloader_Y)

            images_X, labels_X = next(iter_X)
            images_X, labels_X = (
                (images_X).to(device),
                (labels_X).to(device).long().squeeze(),
            )

            images_Y, labels_Y = next(iter_Y)
            images_Y, labels_Y = (
                (images_Y).to(device),
                (labels_Y).to(device).long().squeeze(),
            )

            # ============================================
            #            TRAIN THE DISCRIMINATORS
            # ============================================

            # Train with real images
            d_optimizer.zero_grad()

            # 1. Compute the discriminator losses on real images
            D_X_loss = torch.mean((D_X(images_X) - 1) ** 2)
            D_Y_loss = torch.mean((D_Y(images_Y) - 1) ** 2)

            d_real_loss = D_X_loss + D_Y_loss
            d_real_loss.backward()
            d_optimizer.step()

            # Train with fake images
            d_optimizer.zero_grad()

            # 2. Generate fake images that look like domain X based on real images in domain Y
            fake_X = G_YtoX(images_Y)

            # 3. Compute the loss for D_X
            D_X_loss = torch.mean((D_Y(fake_X)) ** 2)

            # 4. Generate fake images that look like domain Y based on real images in domain X
            fake_Y = G_XtoY(images_X)

            # 5. Compute the loss for D_Y
            D_Y_loss = torch.mean((D_X(fake_Y)) ** 2)

            d_fake_loss = D_X_loss + D_Y_loss
            d_fake_loss.backward()
            d_optimizer.step()

            # =========================================
            #            TRAIN THE GENERATORS
            # =========================================
            g_optimizer.zero_grad()

            # 1. Generate fake images that look like domain X based on real images in domain Y
            fake_X = G_YtoX(images_Y)

            # 2. Compute the generator loss based on domain X
            g_loss = torch.mean((D_X(fake_X) - 1) ** 2)

            reconstructed_Y = G_XtoY(fake_X)
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.mean(
                torch.sum(torch.abs(images_Y - reconstructed_Y), (1, 2, 3))
            )

            g_loss += training_params.lambda_cycle * cycle_consistency_loss

            g_loss.backward()
            g_optimizer.step()

            #########################################
            ##    FILL THIS IN: X--Y-->X CYCLE     ##
            #########################################

            g_optimizer.zero_grad()

            # 1. Generate fake images that look like domain Y based on real images in domain X
            fake_Y = G_XtoY(images_X)

            # 2. Compute the generator loss based on domain Y
            g_loss = torch.mean((D_Y(fake_Y) - 1) ** 2)

            reconstructed_X = G_YtoX(fake_Y)
            # 3. Compute the cycle consistency loss (the reconstruction loss)
            cycle_consistency_loss = torch.mean(
                torch.sum(torch.abs(images_X - reconstructed_X), (1, 2, 3))
            )

            g_loss += training_params.lambda_cycle * cycle_consistency_loss

            g_loss.backward()
            g_optimizer.step()

            # Print the log info
            if iteration % training_params.log_step == 0:
                print(
                    "Iteration [{:5d}/{:5d}] | d_real_loss: {:6.4f} | d_Y_loss: {:6.4f} | d_X_loss: {:6.4f} | "
                    "d_fake_loss: {:6.4f} | g_loss: {:6.4f}".format(
                        iteration,
                        training_params.train_iters,
                        d_real_loss.item(),
                        D_Y_loss.item(),
                        D_X_loss.item(),
                        d_fake_loss.item(),
                        g_loss.item(),
                    )
                )

            # Save the generated samples
            if iteration % training_params.sample_every == 0:
                cyclegan_save_samples(
                    iteration,
                    fixed_Y,
                    fixed_X,
                    G_YtoX,
                    G_XtoY,
                    training_params=training_params,
                    device=device,
                )

            # Save the model parameters
            if iteration % training_params.checkpoint_every == 0:
                cyclegan_checkpoint(
                    iteration, G_XtoY, G_YtoX, D_X, D_Y, training_params
                )

    except KeyboardInterrupt:
        print("Exiting early from training.")
        return {
            "generatorX-Y": G_XtoY,
            "generatorY-X": G_YtoX,
            "discriminatorX": D_X,
            "discriminatorY": D_Y,
        }

    return {
        "generatorX-Y": G_XtoY,
        "generatorY-X": G_YtoX,
        "discriminatorX": D_X,
        "discriminatorY": D_Y,
    }
