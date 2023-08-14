import torch
import numpy as np


print("PyTorch Version:", torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if (not torch.cuda.is_available()):
	print("Warning: Cuda not available, using CPU instead.")

# Use NumPy to create a 2D array of complex numbers on [-1.6,1.6]x[-1.3,1.3]
Y, X = np.mgrid[-1.3:1.3:0.0005, -1.6:1.6:0.0005]

cReal = np.full(X.shape, -1)
cImag = np.full(X.shape, 0)

# load into PyTorch tensors
x  = torch.Tensor(X)
y  = torch.Tensor(Y)
zs = torch.complex(x, y) #important!
ns = torch.zeros_like(zs)
c  = torch.complex(
	torch.Tensor(cReal),
	torch.Tensor(cImag)
)

# transfer to the GPU device
zs = zs.to(device)
ns = ns.to(device)
c = c.to(device)

#Mandelbrot Set
for i in range(200):
	# Compute the new values of z: z^2 + c
	# c is a constant, z is initialised as
	# the value of the starting point
	# If the zs sequences don't diverge for
	# that starting point, that starting point
	# is in the Julia set for c.
	zs_ = zs * zs + c
	#Have we diverged with this new value?
	not_diverged = torch.abs(zs_) < 32.0
	#Update variables to compute
	ns += not_diverged
	zs = zs_

#plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,10))
def processFractal(a):
	"""Display an array of iteration counts as a
	colorful picture of a fractal."""
	a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
	img = np.concatenate([10+20*np.cos(a_cyclic),
	30+50*np.sin(a_cyclic),
	155-80*np.cos(a_cyclic)], 2)
	img[a==a.max()] = 0
	a = img
	a = np.uint8(np.clip(a, 0, 255))
	return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()