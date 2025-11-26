import torch
import torch.nn as nn

class GeometryTable(nn.Module):
    def __init__(self, geometry: torch.Tensor):
        # geometry is a tensor of shape N,2 representing signal I and Q mappings for a corresponding Symbol
        # if the tensor is complex then shape is N,1
        super(GeometryTable, self).__init__()
        self.geometry = nn.Parameter(geometry)
        self.rev_geometry = {k: i for i,k in enumerate(geometry)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signal = self.geometry[x]
        return signal

    def rev_lookup(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # X is shape (B, *, E)
            # Emb is shape (K, E)
            shape = (1,) * (x.unsqueeze(-2).dim() - 2) + self.geometry.size()
            dist = ((x.unsqueeze(-2) - self.geometry.view(shape)) *
                    (x.unsqueeze(-2) - self.geometry.view(shape)).conj()).sum(-1).real
            ind = dist.min(dim=-1).indices
        return ind
