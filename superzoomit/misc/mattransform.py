import numpy as np

def generate_and_flatten_grid(v1, v2=None, grid='X'):
    """Generates a mesh grid and then vectorize it, concatenating the rows"""
    if v2 is None:
        v2 = v1
    if (grid == 'Y') | (grid == 'y'):
        grid = 1
    else:
        grid = 0
    tmp_grid = np.meshgrid(v1, v2)[grid]
    return vectorize_matrix(tmp_grid.T)

def vectorize_matrix(matrix):
    """Transforms a matrix into a vector, by concatenating the rows. Takes a numpy ndarray as input"""
    n, m = matrix.shape
    return matrix.reshape(n * m)

