import math
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from model.Decoder import ResidualDecoder
from model.Encoder import ResidualEncoder
from model.GeometryTable import GeometryTable
from model.VQVAE import VQVAE
from model.VectorQuantization import VectorQuantizer

img_to_tensor = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
])
dataset = torchvision.datasets.CIFAR10(root="data/", transform=img_to_tensor, train=True, download=False)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(0))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 64
encoder = ResidualEncoder(64, 2 , 256)
decoder = ResidualDecoder(64, 2 , 256)
rectangular_map = torch.cartesian_prod(
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K))),
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K)))
)
geometry = GeometryTable(rectangular_map)
quantizer = VectorQuantizer(K, 256)
model = VQVAE(encoder, decoder, quantizer)
model.to(device)
mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

val_loss = float('inf')
patience = 5
count = 1
for epoch in range(40):
    total_loss = {
        "recon_loss": 0,
        "embedding_loss": 0,
        "val_loss": 0,
    }
    for images, _ in train_loader:
        images = images.to(device)
        z_e, z_q, x_hat = model(images)
        recon_loss = mse(images, x_hat)
        embedding_loss = mse(z_e, z_q)
        loss =  recon_loss + 0.1 * embedding_loss
        total_loss["recon_loss"] += recon_loss.item()
        total_loss["embedding_loss"] += embedding_loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            _, _, x_hat = model(images)
            total_loss["val_loss"] += mse(images, x_hat).item()
    if val_loss > total_loss["val_loss"]:
        val_loss = total_loss["val_loss"]
        count = 1
    else:
        if count >= patience:
            break
        else:
            count += 1
    print(f"Epoch {epoch+1:3d} | Reconstruction Loss: {total_loss["recon_loss"]/len(train_loader):.4f} "
          f"| Embedding Loss: {total_loss["embedding_loss"]/len(train_loader):.4f} "
          f"| Validation Loss: {total_loss['val_loss']/len(val_loader):.4f}")
torch.save(model.state_dict(), 'saved_models/model.pth')