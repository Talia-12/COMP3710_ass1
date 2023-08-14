import torch
import numpy as np

print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	print("Warning: Cuda not available, using CPU instead.")

# parameters
amplitude = 5
frequency = 0.5
phase     = 0
direction = np.pi / 6

dirX = np.cos(direction)
dirY = np.sin(direction)

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
trigFun = amplitude * torch.sin(frequency * (2*np.pi) * (x * dirX + y * dirY - phase))
zs = [
  torch.exp(-(x**2+y**2)/2.0),
  trigFun,
  torch.exp(-(x**2+y**2)/2.0) * trigFun
]

#plot
import matplotlib.pyplot as plt
for z in zs:
	plt.imshow(z.cpu().numpy(), extent=[-4.0, 4.0, -4.0, 4.0])
	plt.tight_layout()
	plt.show()