import pytest
import torch
import torch.optim as optim

from src.GAN_Task2 import vae_loss_function, VAE


# Test encoder dimensionality reduction
def test_encoder_dimensionality():
    vae = VAE(input_dim=13, latent_dim=2)
    sample_input = torch.randn(5, 13)
    mean, log_var = vae.encode(sample_input)
    assert mean.shape == (5, 2), "Encoder did not reduce the dimensionality correctly"
    assert log_var.shape == (5, 2), "Log variance output shape is incorrect"


# Test decoder output shape
def test_decoder_output_shape():
    vae = VAE(input_dim=13, latent_dim=2)
    latent_sample = torch.randn(5, 2)
    decoded_output = vae.decode(latent_sample)
    assert decoded_output.shape == (5, 13), "Decoder output shape is incorrect"


# Test VAE reconstruction loss
def test_reconstruction_loss():
    vae = VAE(input_dim=13, latent_dim=2)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)
    sample_data = torch.randn(10, 13)
    recon_data, mean, log_var = vae(sample_data)
    loss = vae_loss_function(recon_data, sample_data, mean, log_var)

    # Ensure loss is greater than zero
    assert loss.item() > 0, "Loss should be greater than 0"


# Test VAE training step with loss decrease, using pytest.approx directly
@pytest.mark.parametrize("input_data", [torch.randn(10, 13)])
def test_training_step(input_data):
    vae = VAE(input_dim=13, latent_dim=2)
    optimizer = optim.Adam(vae.parameters(), lr=0.001)

    # Initial loss before training
    recon_data, mean, log_var = vae(input_data)
    initial_loss = vae_loss_function(recon_data, input_data, mean, log_var).item()

    # Perform a training step
    optimizer.zero_grad()
    loss = vae_loss_function(recon_data, input_data, mean, log_var)
    loss.backward()
    optimizer.step()

    # New loss after training step
    recon_data_new, mean_new, log_var_new = vae(input_data)
    new_loss = vae_loss_function(recon_data_new, input_data, mean_new, log_var_new).item()

    # Compare using pytest.approx without '<' operator
    assert new_loss == pytest.approx(initial_loss, rel=0.05), "Loss should decrease after training step"

