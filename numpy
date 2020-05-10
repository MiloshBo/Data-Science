import numpy as np 
import array

print(np.__version__)

L = list(range(10))
A = array.array('i', L)

print(np.array([1, 4, 2, 5, 3]))
print(np.array([3.14, 4, 2, 3]))
print(np.array([1, 2, 3, 4], dtype='float32'))

np.array([range(i, i + 3) for i in [2, 4, 6]])
np.zeros(10, dtype=int)
np.ones((3, 5), dtype=float)
np.full((3,5), 3.14)
np.arange(0, 20, 2)
np.linspace(0, 1, 5)
np.random.random((3, 3))
np.random.normal(0, 1, (3, 3))
np.random.randint(0, 10, (3, 3))
np.eye(3)
np.empty(3)
np.zeros(10, dtype='int16')
np.zeros(10, dtype=np.int16)

array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int16)

import numpy as np
np.random.seed(0)
x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(3, 4))
x3 = np.random.randint(10, size=(3, 4, 5))
print(x1)
print(x2)
print(x3)
print("x3 ndim: ", x3.ndim)
print("x3 shape: ", x3.shape)
print("x3 size: ", x3.size)
print("dtype: ", x3.dtype)
print("itemsize: ", x3.itemsize, "bytes")
print("nbytes: ", x3.nbytes, "bytes")
print(x1[4])
print(x2[2, 0])

to modife values:
x2[0, 0] = 12
print(x2)


array slicing:
x[start:stop:step]
x[::-1]      # reverse array
x[5::-2]     # reversed every other from index 5

Multi-dimensional subarrays
x2[:2, :3]        # two rows, three columns
x2[:3, ::2]       # all rows, every other column
x2[::-1, ::-1]    # reversed
print(x2[:, 0])   # first column of x2
print(x2[0, :])   # first row of x2
print(x2[0])      # equivalent to x2[0, :]

x2_sub_copy = x2[:2, :2].copy()      # Creating copies of arrays

grid = np.arange(1, 10).reshape((3, 3))      # Reshaping of arrays
x.reshape((1, 3))         # row vector via reshape
x[np.newaxis, :]          # row vector via newaxis
x.reshape((3, 1))         # column vector via reshape
x[:, np.newaxis]          # column vector via newaxis

x = np.array([1, 2, 3])
y = np.array([3, 2, 1])
z = [99, 99, 99]
np.concatenate([x, y, z])   # concatenation

grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
np.concatenate([grid, grid])   # concatenate along the first axis
np.concatenate([grid, grid], axis=1)

# working with arrays of mixed dimensions
x = np.array([1, 2, 3])
grid = np.array([[9, 8, 7],
                 [6, 5, 4]])
np.vstack([x, grid])

# horizontally stack the arrays
y = np.array([[99],
              [99]])
np.hstack([grid, y])

np.dstack  # 3rd stack array

# splitting of arrays
x = [1, 2, 3, 99, 99, 3, 2, 1]
x1, x2, x3 = np.split(x, [3, 5])

grid = np.arange(16).reshape((4, 4))
upper, lower = np.vsplit(grid, [2])
left, right = np.hsplit(grid, [2])
