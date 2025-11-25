import torch
import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, quantizer):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    def encode(self, x):
        h = self.encoder(x)
        z_q = self.quantizer.quantize(h.transpose(1, 2).transpose(2, 3))
        return z_q

    def decode(self, z_q):
        z_q = self.quantizer.ind_to_embeddings(z_q)
        x_hat = self.decoder(z_q.transpose(2, 3).transpose(1, 2))
        return x_hat

    def forward(self, x):
        h = self.encoder(x)
        z_q, emb = self.quantizer(h.transpose(1, 2).transpose(2, 3))
        x_hat = self.decoder(z_q.transpose(2,3).transpose(1,2))
        return h, emb.transpose(2,3).transpose(1,2), x_hat