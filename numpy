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
