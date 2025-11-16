from model.VectorQuantization import VectorQuantizer
from copy import deepcopy
import torch

print("Test embedding gradients are zero and straight through to z_q")
vq = VectorQuantizer(4,5)
x = torch.randn(2,5,5)
x.requires_grad_()
y = torch.randn(2,5,5)
emb1, emb2 = vq(x)
((emb1-y)**2).sum().backward()
# Pass Through Gradients, Embedding has no gradients
print(x.grad)
print(vq.embeddings.grad)
print("-----------------------------------------------------------------------------------")
# Verify embeddings still have gradients
print("Embedding still has gradients if not detached")
print(emb2)
print("-----------------------------------------------------------------------------------")
# Copy gradients to verify for test 2
grads = deepcopy(x.grad)
# Zero gradients before next text
x.grad.zero_()

# Verify that straight through gradients when combined match emb gradients
# our input will not have any gradients due to non-differentiable lookup
print("Test that straight through gradients sum by index are correct")
print("Indices from quantization and gradients for z_q from earlier")
ind = vq.quantize(x)
emb = vq.embeddings[ind]
emb_grads = torch.zeros_like(vq.embeddings)
((emb-y)**2).sum().backward()
print(ind)
print(grads)
print("-----------------------------------------------------------------------------------")

print("Testing that gradients are zero because non-differentiable lookup")
print(x.grad)
print("-----------------------------------------------------------------------------------")

count = 0
ind = ind.flatten()
for i in grads.view(-1,5):
    emb_grads[ind[count]] += i
    count += 1
print("Test embedding gradients = sum of z_q gradients for each index")
print(vq.embeddings.grad)
print(emb_grads)
print("-----------------------------------------------------------------------------------")
vq.embeddings.grad.zero_()

print("Test it works for any input size")
print("Test with 3,4,7")
x = torch.randn(3,4,7,5)
print(vq.get_embeddings(x).shape)
print("-----------------------------------------------------------------------------------")

print("Test with 128,10,9,5,11")
x = torch.randn(128,10,9,5,11,5)
print(vq.get_embeddings(x).shape)
print("-----------------------------------------------------------------------------------")