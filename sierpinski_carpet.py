import torch
import numpy as np


print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	print("Warning: Cuda not available, using CPU instead.")

layers = 8

# Use NumPy to create a 2D array of numbers on [-2,1]x[-1.3,1.3]
Y, X = np.mgrid[0:3**layers:1, 0:3**layers:1]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

out = torch.zeros_like(x)

for layer in range(1, layers + 1):
	outLayer = torch.eq(x, x)
	for dir in (x,y):
		layerScaledIndex = torch.div(torch.remainder(dir, 3**layer), 3**(layer-1), rounding_mode="floor")	
		shouldBeFilled = torch.eq(layerScaledIndex, 1)
		outLayer = torch.logical_and(outLayer, shouldBeFilled)
	out = torch.logical_or(out, outLayer)

print(out.cpu().numpy())

#plot
import matplotlib.pyplot as plt
plt.imshow(out.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
plt.tight_layout()
plt.show()