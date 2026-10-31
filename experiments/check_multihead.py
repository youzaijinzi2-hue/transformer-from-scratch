
import torch

from transformer.attention import MultiHeadAttention

torch.manual_seed(42)

B = 2
N = 4
D = 8
H = 2

X = torch.randn(B, N, D)

mha = MultiHeadAttention(
    d_model=D,
    num_heads=H
)

Z, A = mha(X)

print("Input:", X.shape)
print("Output:", Z.shape)
print("Attention weights:", A.shape)

print("Head 1:")
print(A[0, 0])

print("Head 2:")
print(A[0, 1])
