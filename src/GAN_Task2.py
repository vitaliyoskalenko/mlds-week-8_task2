import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os

torch.manual_seed(42)

# Load Wine dataset
data = load_wine()
X = data['data']  # Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# Dataset and Dataloader
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, 64)
        self.encoder_fc2 = nn.Linear(64, 32)
        self.fc_mean = nn.Linear(32, latent_dim)
        self.fc_log_var = nn.Linear(32, latent_dim)

        # Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, 32)
        self.decoder_fc2 = nn.Linear(32, 64)
        self.decoder_fc3 = nn.Linear(64, input_dim)
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.encoder_fc1(x))
        h = self.relu(self.encoder_fc2(h))
        mean = self.fc_mean(h)
        log_var = self.fc_log_var(h)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        h = self.relu(self.decoder_fc1(z))
        h = self.relu(self.decoder_fc2(h))
        return self.decoder_fc3(h)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var


# Loss function (Reconstruction + KL Divergence)
def vae_loss_function(recon_x, x, mean, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return recon_loss + kl_divergence


# Hyperparameters
input_dim = X.shape[1]  # 13 features
latent_dim = 2  # Latent space dimension
epochs = 50
learning_rate = 0.001

# Initialize VAE and optimizer
vae = VAE(input_dim=input_dim, latent_dim=latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Training loop
losses = []
vae.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in dataloader:
        x_batch = batch[0]
        optimizer.zero_grad()
        recon_batch, mean, log_var = vae(x_batch)
        loss = vae_loss_function(recon_batch, x_batch, mean, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    losses.append(total_loss / len(dataloader.dataset))
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {losses[-1]}")

# Evaluate reconstruction
vae.eval()
with torch.no_grad():
    sample_data = X_tensor[:10]  # Select a small batch of data
    recon_data, _, _ = vae(sample_data)

# Compare original and reconstructed data
print("Original Data:\n", sample_data.numpy())
print("\nReconstructed Data:\n", recon_data.numpy())


if __name__ == "__main__":
    os.makedirs('docs', exist_ok=True)
    plt.rcParams.update({
        'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
        'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': '#52514e',
        'xtick.color': '#52514e', 'ytick.color': '#52514e', 'text.color': '#0b0b0b',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': '#e8e7e3', 'grid.linewidth': 0.8,
    })

    # Training loss per epoch
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), losses, color='#2a78d6', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss per sample (MSE + KL)')
    plt.title('VAE training loss', loc='left')
    plt.tight_layout()
    plt.savefig('docs/loss.png', dpi=120)
    plt.show()

    # 2D latent space (encoder mean) coloured by wine class
    with torch.no_grad():
        latent_mean, _ = vae.encode(X_tensor)
    latent_mean = latent_mean.numpy()
    colors = ['#2a78d6', '#eb6834', '#1baf7a']
    plt.figure(figsize=(8, 6))
    for label, (name, color) in enumerate(zip(data['target_names'], colors)):
        points = latent_mean[data['target'] == label]
        plt.scatter(points[:, 0], points[:, 1], s=40, color=color,
                    edgecolors='#fcfcfb', linewidths=1, label=name)
        cx, cy = points.mean(axis=0)
        plt.annotate(name, (cx, cy), fontsize=11, fontweight='bold', color='#0b0b0b',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', fc='#fcfcfb', ec=color, alpha=0.9))
    plt.xlabel('Latent dimension 1')
    plt.ylabel('Latent dimension 2')
    plt.title('Wine dataset in the 2D latent space', loc='left')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig('docs/latent_space.png', dpi=120)
    plt.show()

    # Original vs reconstructed data (standardized features)
    limit = np.abs(np.concatenate([sample_data.numpy(), recon_data.numpy()])).max()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, values, title in zip(axes, [sample_data.numpy(), recon_data.numpy()],
                                 ['Original', 'Reconstructed']):
        image = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        ax.set_title(title, loc='left')
        ax.set_xticks(range(input_dim))
        ax.set_xticklabels(data['feature_names'], rotation=60, ha='right', fontsize=8)
        ax.set_yticks(range(len(values)))
        ax.grid(False)
    axes[0].set_ylabel('Sample')
    fig.colorbar(image, ax=axes, label='Standardized value')
    fig.suptitle('Original vs reconstructed data (first 10 samples)', x=0.02, ha='left')
    plt.savefig('docs/reconstruction.png', dpi=120, bbox_inches='tight')
    plt.show()
