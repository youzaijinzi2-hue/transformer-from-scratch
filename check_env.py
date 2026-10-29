
import torch

print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)

x = torch.randn(3, 4, device=device)
w = torch.randn(4, 5, device=device, requires_grad=True)

y = x @ w
loss = (y ** 2).mean()
loss.backward()

print("Output shape:", y.shape)
print("Gradient available:", w.grad is not None)
