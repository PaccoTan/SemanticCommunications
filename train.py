import math
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from channels.AWGN import AWGN
from model.Decoder import ResidualDecoder
from model.Encoder import ResidualEncoder
from model.GeometryTable import GeometryTable
from model.VQVAE import VQVAE
from model.VectorQuantization import VectorQuantizer
from pytorch_msssim import ssim

img_to_tensor = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
])
dataset = torchvision.datasets.CIFAR10(root="data/", transform=img_to_tensor, train=True, download=False)
test_dataset = torchvision.datasets.CIFAR10(root="data/", train=False, download=False)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(0))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params_list = []
K_set = [4, 16, 64]
SNR_set = [0, 5, 10]
modulation_set = ["QAM", "PSK"]
emb_dims = [64,128,256]

for i in range(len(K_set)):
    for snr in SNR_set:
        for modul in modulation_set:
            for emb_dim in emb_dims:
                params = {
                    "K": K_set[i],
                    "SNR": snr + i * 5,
                    "modul": modul,
                    "emb_dim": emb_dim,
                }
                params_list.append(params)

for param in params_list:
    # Hyper Parameters
    K = param["K"]
    SNR = param["SNR"]
    modulation = param["modul"]
    num_channels = 64
    depth = 2
    emb_dim = param["emb_dim"]

    # Define Modulation and Channel Scheme
    match modulation:
        case "PSK":
            psk = torch.arange(0, K) * 2 * torch.pi / K
            psk = torch.stack([psk.sin(), psk.cos()], -1)
            symbol_energy = (psk ** 2).sum(-1).mean()
            geometry = GeometryTable(psk)
        case "QAM":
            rectangular_map = torch.cartesian_prod(
                torch.linspace(int(-math.sqrt(K) + 1), int(math.sqrt(K) - 1), int(math.sqrt(K))),
                torch.linspace(int(-math.sqrt(K) + 1), int(math.sqrt(K) - 1), int(math.sqrt(K)))
            )
            symbol_energy = (rectangular_map ** 2).sum(-1).mean()
            geometry = GeometryTable(rectangular_map)
        case _:
            modulation = "QAM"
            rectangular_map = torch.cartesian_prod(
                torch.linspace(int(-math.sqrt(K) + 1), int(math.sqrt(K) - 1), int(math.sqrt(K))),
                torch.linspace(int(-math.sqrt(K) + 1), int(math.sqrt(K) - 1), int(math.sqrt(K)))
            )
            symbol_energy = (rectangular_map ** 2).sum(-1).mean()
            geometry = GeometryTable(rectangular_map)
    geometry.to(device)
    variance = symbol_energy / torch.pow(10, torch.Tensor([SNR/10]))
    variance = variance.item()
    channel = AWGN(variance)


    # Define Model
    encoder = ResidualEncoder(num_channels, depth, emb_dim)
    decoder = ResidualDecoder(num_channels, depth, emb_dim)
    quantizer = VectorQuantizer(K, emb_dim)
    model = VQVAE(encoder, decoder, quantizer)
    model.to(device)

    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    val_loss = float('inf')
    patience = 5
    count = 1
    print(" Epoch |  Recons |  Embed   | Val Loss |  SSIM  ")
    for epoch in range(40):
        total_loss = {
            "recon_loss": 0,
            "embedding_loss": 0,
            "val_loss": 0,
            "ssim_loss": 0,
        }
        for images, _ in train_loader:
            images = images.to(device)
            # get quantized representation of images
            # add noise and retrieve alphabet correspondence
            # retrieve processed noisy embeddings
            z_e = model.encoder(images)
            z_q = geometry(model.quantizer.quantize(z_e.transpose(1, 2).transpose(2, 3)))
            z_q = geometry.rev_lookup(channel(z_q))
            emb = model.quantizer.ind_to_embeddings(z_q).transpose(2,3).transpose(1,2)
            z_q = z_e + (emb - z_e).detach()
            x_hat = model.decoder(z_q)
            recon_loss = mse(images, x_hat)
            embedding_loss = mse(z_e, emb)
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
                total_loss["ssim_loss"] += ssim(images, x_hat, data_range=1.0)
        if val_loss > total_loss["val_loss"]:
            val_loss = total_loss["val_loss"]
            count = 1
        else:
            if count >= patience:
                break
            else:
                count += 1
        print(f"   {epoch+1:3d} |  {total_loss["recon_loss"]/len(train_loader):.4f} "
              f"|  {total_loss["embedding_loss"]/len(train_loader):.4f}  "
              f"|  {total_loss['val_loss']/len(val_loader):.4f}  "
              f"| {total_loss['ssim_loss']/len(val_loader):.4f}")
    torch.save(model.state_dict(), f'saved_models/model_{K}{modulation}_{SNR}_{emb_dim}.pth')