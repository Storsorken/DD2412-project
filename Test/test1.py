import numpy as np

A = np.array([[1, 2, 3, 2],
            [4, 5, 6, 4],
            [7, 8, 9, 1]])

# Create a 3D array of shape (n, 4, 4) indicating whether pairs are equal
equality_array = (A[:, :, None] == A[:, None, :]).astype(int)

print(equality_array.sum(axis = 0))