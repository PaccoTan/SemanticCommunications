from model.GeometryTable import GeometryTable
import torch
import math
import matplotlib.pyplot as plt

K = 4
x = torch.randint(0,K,(1000,5))
std = math.sqrt(0.5)
rectangular_map = torch.cartesian_prod(
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K))),
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K)))
)
plt.scatter(rectangular_map[:,0].numpy(), rectangular_map[:,1].numpy(),c=[0,1,2,3])
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.title(f"Constellation Diagram Example for {K}-QAM")
plt.show()
plt.close()

geometry = GeometryTable(rectangular_map)
mapped_signal = geometry(x)
noise = torch.randn(mapped_signal.shape)*std
awgn_signal = mapped_signal + noise
noisy_points = awgn_signal.view(-1,2)
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

received_symbol = geometry.rev_lookup(awgn_signal)
print(f"{K}-QAM with AWGN Channel STD={std:.4f}")
print("Number of symbols sent:",x.numel())
print(f"Correct Received Symbols: {((received_symbol == x).sum()/x.numel()).item()*100:.2f}%")
print("-------------------------------------------------------------------------")



psk = torch.arange(0,K)*2*torch.pi/K
psk = torch.stack([psk.sin(),psk.cos()],-1)
plt.scatter(psk[:,0].numpy(), psk[:,1].numpy(),c=[0,1,2,3])
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.title("Constellation Diagram Example for PSK")
plt.show()
plt.close()

geometry = GeometryTable(psk)
mapped_signal = geometry(x)
noise = torch.randn(mapped_signal.shape)*std
awgn_signal = mapped_signal + noise
noisy_points = awgn_signal.view(-1,2)
plt.scatter(psk[:,0].numpy(), psk[:,1].numpy(),c='red',s=60,zorder=3)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.title("PSK Received Symbols with AWGN")
scatter = plt.scatter(noisy_points[:,0].numpy(), noisy_points[:,1].numpy(),c=x.view(-1).numpy(),marker='x',alpha=0.5)
cbar = plt.colorbar(scatter, ticks=range(len(x.unique())))
cbar.set_label('Symbol Label')
plt.show()
plt.close()
received_symbol = geometry.rev_lookup(awgn_signal)
print(f"{K}-PSK with AWGN Channel STD={std:.4f}")
print("Number of symbols sent:",x.numel())
print(f"Correct Received Symbols: {((received_symbol == x).sum()/x.numel()).item()*100:.2f}%")
print("-------------------------------------------------------------------------")



rectangular_map = torch.cartesian_prod(
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K))),
    torch.linspace(int(-math.sqrt(K)+1),int(math.sqrt(K)-1),int(math.sqrt(K)))
)
rectangular_map = torch.view_as_complex(rectangular_map)
plt.scatter(rectangular_map.real.numpy(), rectangular_map.imag.numpy(),c=[0,1,2,3])
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.title(f"Constellation Diagram Example for {K}-QAM")
plt.show()
plt.close()

geometry = GeometryTable(rectangular_map.view(-1,1))
mapped_signal = geometry(x)
noise = torch.view_as_complex(torch.stack([torch.randn(mapped_signal.shape)*std,torch.randn(mapped_signal.shape)*std],-1))
awgn_signal = mapped_signal + noise
noisy_points = awgn_signal.view(-1)
plt.scatter(rectangular_map.real.numpy(), rectangular_map.imag.numpy(),c='red',s=60,zorder=3)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle='--', alpha=0.5)
plt.title(f"{K}-QAM Received Symbols with AWGN")
scatter = plt.scatter(noisy_points.real.numpy(), noisy_points.imag.numpy(),c=x.view(-1).numpy(),marker='x',alpha=0.5)
cbar = plt.colorbar(scatter, ticks=range(len(x.unique())))
cbar.set_label('Symbol Label')
plt.show()
plt.close()

received_symbol = geometry.rev_lookup(awgn_signal)
print(f"{K}-QAM with AWGN Channel STD={std:.4f}")
print("Number of symbols sent:",x.numel())
print(f"Correct Received Symbols: {((received_symbol == x).sum()/x.numel()).item()*100:.2f}%")
print("-------------------------------------------------------------------------")