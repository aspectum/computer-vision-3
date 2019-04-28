import numpy as np

N = 3

W = 15
x, y = 15, 15

data = np.random.randint(20, size=(30, 30))
current_patch = np.random.randint(20, size=(N, N))

# isolate the patch
x_start = x - W//2
y_start = y - W//2

patch = data[x_start:x_start+W, y_start:y_start+W]

# take a windowed view of the array
from numpy.lib.stride_tricks import as_strided
shape = tuple(np.subtract(patch.shape, N-1)) + (N, N)
windowed_patch = as_strided(patch, shape=shape, strides=patch.strides*2)

# this works, but creates a large intermediate array
cost = np.abs(windowed_patch - current_patch).sum(axis=(-1, -2))

# this is much more memory efficient, but uses squared differences,
# and the fact that (a - b)^2 = a^2 + b^2 - 2*a*b
patch2 = patch*patch
ssqd =  as_strided(patch2, shape=shape,
                   strides=patch2.strides*2).sum(axis=(-1, -2),
                                                 dtype=np.double)
ssqd += np.sum(current_patch * current_patch, dtype=np.double)
ssqd -= 2 * np.einsum('ijkl,kl->ij', windowed_patch, current_patch,
                      dtype=np.double)

# for comparison with the other method
cost2 = windowed_patch - current_patch
cost2 = np.sum(cost2*cost2, axis=(-1, -2))

# with any method, to find the location of the max in cost:
best_X, best_y = np.unravel_index(np.argmax(cost), cost.shape)