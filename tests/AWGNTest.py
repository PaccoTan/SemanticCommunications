import torch
from matplotlib import pyplot as plt

from channels.AWGN import AWGN
import math
from model.GeometryTable import GeometryTable


K = 4
rectangular_map = torch.cartesian_prod(
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K))),
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K)))
)
geometry = GeometryTable(rectangular_map)

n = torch.distributions.normal.Normal(0,1)
signal_error = torch.Tensor([0.1,0.01,0.001,0.0001])
symbol_energy = (rectangular_map ** 2).sum(-1).sqrt().mean()

with torch.no_grad():
    x = torch.randint(0,K,(10000,5,5))
    for error in signal_error:
        variance = 1/(n.icdf(1-torch.sqrt(1-error))**2 * 0.5)
        noise = AWGN(variance.sqrt().item())
        noisy_x = noise(geometry(x),"cpu")
        noisy_points = noisy_x.view(-1,2)
        plt.scatter(rectangular_map[:,0].numpy(), rectangular_map[:,1].numpy(),c='red',s=60,zorder=3)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title(f"{K}-QAM Received Symbols with AWGN")
        scatter = plt.scatter(noisy_points[:,0].numpy(), noisy_points[:,1].numpy(),c=x.view(-1).numpy(),marker='x',alpha=0.5)
        cbar = plt.colorbar(scatter, ticks=range(len(x.unique())))
        cbar.set_label('Symbol Label')
        plt.show()
        plt.close()
        received_symbol = geometry.rev_lookup(noisy_x)
        print(f"{K}-QAM with AWGN Channel Variance = {variance:.4f}")
        print(f"Expected Accuracy: {((1-error)*100):.3f}%")
        print(f"Actual Accuracy: {(received_symbol == x).sum()}/{x.numel()} = {((received_symbol == x).sum()/x.numel())*100:.2f}%")
        print(f"SNR: {10*torch.log10(symbol_energy/variance):.2f}dB")
        print("-------------------------------------------------------------------------")

    print("\n\n\n\nResults using Complex Representation")
    geometry = GeometryTable(torch.view_as_complex(rectangular_map).unsqueeze(-1))
    for error in signal_error:
        variance = 1/(n.icdf(1-torch.sqrt(1-error))**2 * 0.5)
        noise = AWGN(variance.sqrt().item())
        noisy_x = noise(geometry(x),"cpu")
        received_symbol = geometry.rev_lookup(noisy_x)
        print(f"{K}-QAM with AWGN Channel Variance = {variance:.4f}")
        print(f"Expected Accuracy: {((1-error)*100):.3f}%")
        print(f"Actual Accuracy: {(received_symbol == x).sum()}/{x.numel()} = {((received_symbol == x).sum()/x.numel())*100:.2f}%")
        print(f"SNR: {10*torch.log10(symbol_energy/variance):.2f}dB")
        print("-------------------------------------------------------------------------")

K = 64
rectangular_map = torch.cartesian_prod(
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K))),
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K)))
)
geometry = GeometryTable(rectangular_map)

n = torch.distributions.normal.Normal(0,1)
variance = torch.Tensor([0.5,0.75,1])
symbol_energy = (rectangular_map ** 2).sum(-1).mean()
print(f"\n\n\n\nResults for K={K}")
with torch.no_grad():
    x = torch.randint(0,K,[100000])
    for variance in variance:
        factor = 4 * (1-1/math.sqrt(K))
        p_error = factor * (1-n.cdf((3*symbol_energy/variance/(K-1)).sqrt())
                            - (1-1/math.sqrt(K))*(1-n.cdf((3*symbol_energy/variance/(K-1)).sqrt()))**2)
        noise = AWGN(variance.sqrt().item())
        noisy_x = noise(geometry(x), "cpu")
        received_symbol = geometry.rev_lookup(noisy_x)
        print(f"{K}-QAM with AWGN Channel Variance = {variance:.4f}")
        print(f"First Term: {factor * (1-n.cdf((3*symbol_energy/variance/(K-1)).sqrt())):.4f} | "
              f"Second Term: {factor * (1-1/math.sqrt(K)) * (1-n.cdf((3*symbol_energy/variance/(K-1)).sqrt()))**2:.4f}")
        print(f"Average Symbol Energy: {symbol_energy:.2f}")
        print(f"Expected Accuracy: {((1 - p_error) * 100):.3f}%")
        print(f"Actual Accuracy: {(received_symbol == x).sum()}/{x.numel()} = {((received_symbol == x).sum() / x.numel()) * 100:.2f}%")
        print(f"SNR: {10 * torch.log10(symbol_energy / variance):.2f}dB")
        print("-------------------------------------------------------------------------")