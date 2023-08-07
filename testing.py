import torch
import numpy as np

print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	print("Warning: Cuda not available, using CPU instead.")

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# Compute Gaussian
zs = [
  torch.exp(-(x**2+y**2)/2.0),
  torch.sin(x),
  torch.exp(-(x**2+y**2)/2.0) * torch.sin(x)
]

#plot
import matplotlib.pyplot as plt
for z in zs:
	plt.imshow(z.cpu().numpy())
	plt.tight_layout()
	plt.show()