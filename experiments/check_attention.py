import torch

from transformer.attention import scaled_dot_product_attention

torch.manual_seed(42)

B = 2
N = 4
dk = 8
dv = 8

Q = torch.randn(B, N, dk)
K = torch.randn(B, N, dk)
V = torch.randn(B, N, dv)

Z, A = scaled_dot_product_attention(Q, K, V)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)
print("Attention weights:", A.shape)
print("Output:", Z.shape)

# Sum of attention weights along Key dimension
print("Row sums:")
print(A.sum(dim=-1))

causal_mask = torch.tril(
    torch.ones(N, N, dtype=torch.bool)
)

Z_masked, A_masked = scaled_dot_product_attention(
    Q, K, V, mask=causal_mask
)

print("Causal mask:")
print(causal_mask.int())

print("Masked attention weights:")
print(A_masked[0])