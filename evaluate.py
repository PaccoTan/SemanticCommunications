import json
import math
import torch
import matplotlib.pyplot as plt
import torchvision
from pytorch_msssim import ssim
from torchvision.transforms import v2
from channels.AWGN import AWGN
from model.Decoder import ResidualDecoder
from model.Encoder import ResidualEncoder
from model.GeometryTable import GeometryTable
from model.VQVAE import VQVAE
from model.VectorQuantization import VectorQuantizer
from tqdm import tqdm

img_to_tensor = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
])
test_dataset = torchvision.datasets.CIFAR10(root="data/", transform=img_to_tensor, train=False, download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

params_list = []
K_set = [4,16,64]
SNR_set = [0,5,10]
modulation_set = ["QAM"]
emb_dims = [256]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

results = {

}
channel_snr = torch.arange(0,25,1,device=device)
mse = torch.nn.MSELoss()
for param in params_list:
    with torch.no_grad():
        K = param["K"]
        modulation = param["modul"]
        num_channels = 64
        depth = 2
        emb_dim = param["emb_dim"]
        SNR = param["SNR"]
        result_name = f"{K}_{SNR}"
        results[result_name] = {
            "SSIM": torch.zeros_like(channel_snr).float(),
            "MSE": torch.zeros_like(channel_snr).float(),
            "SNR": channel_snr
        }
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
        encoder = ResidualEncoder(num_channels, depth, emb_dim)
        decoder = ResidualDecoder(num_channels, depth, emb_dim)
        quantizer = VectorQuantizer(K, emb_dim)
        model = VQVAE(encoder, decoder, quantizer)
        model_path = f"saved_models/QAM/model_{K}{modulation}_{SNR}_{emb_dim}.pth"
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        for images, _ in tqdm(test_loader):
            count = 0
            images = images.to(device)
            z_e = model.encoder(images)
            z_q = geometry(model.quantizer.quantize(z_e.transpose(1, 2).transpose(2, 3)))
            for snr in channel_snr:
                variance = symbol_energy / torch.pow(10, torch.Tensor([snr / 10]))
                variance = variance.item()
                channel = AWGN(variance)
                noisy_z_q = geometry.rev_lookup(channel(z_q))
                emb = model.quantizer.ind_to_embeddings(noisy_z_q).transpose(2, 3).transpose(1, 2)
                x_hat = model.decoder(emb)
                results[result_name]["SSIM"][count] += ssim(x_hat, images, data_range=1.0)
                results[result_name]["MSE"][count] += mse(x_hat, images)
                count += 1
        results[result_name]["SSIM"] /= len(test_loader)
        results[result_name]["MSE"] /= len(test_loader)
        results[result_name]["SSIM"] = results[result_name]["SSIM"].tolist()
        results[result_name]["MSE"] = results[result_name]["MSE"].tolist()
        results[result_name]["SNR"] = results[result_name]["SNR"].tolist()
json.dump(results, open("results.json", "w"), indent=4)
