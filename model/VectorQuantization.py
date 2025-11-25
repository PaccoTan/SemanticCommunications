import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, k: int = 64, emb_dim: int = 256):
        super(VectorQuantizer, self).__init__()
        self.embeddings = nn.Parameter(torch.empty(k,emb_dim))
        nn.init.kaiming_uniform_(self.embeddings, nonlinearity='relu')

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # X is shape (B, *, E)
            # Emb is shape (K, E)
            shape = (1,) * (x.unsqueeze(-2).dim() - 2) + self.embeddings.size()
            dist = ((x.unsqueeze(-2) - self.embeddings.view(shape)) ** 2).sum(-1)
            ind = dist.min(dim=-1).indices
        return ind

    def ind_to_embeddings(self, ind: torch.Tensor) -> torch.Tensor:
        return self.embeddings[ind]

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        ind = self.quantize(x)
        return self.embeddings[ind]

    def forward(self, x):
        emb = self.get_embeddings(x)
        # return z_q with straight through, and embedding for dictionary updates
        return x + (emb - x).detach(), emb