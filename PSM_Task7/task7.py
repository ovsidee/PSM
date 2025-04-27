import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def build_coefficient_matrix(n: int, T_top, T_bottom, T_left, T_right):
    interior_n = n - 1
    A = lil_matrix((interior_n**2, interior_n**2))
    b = np.zeros(interior_n**2)

    for i in range(interior_n):
        for j in range(interior_n):
            idx = i * interior_n + j
            A[idx, idx] = -4

            if i > 0:
                A[idx, idx - interior_n] = 1
            else:
                b[idx] -= T_top

            if i < interior_n - 1:
                A[idx, idx + interior_n] = 1
            else:
                b[idx] -= T_bottom

            if j > 0:
                A[idx, idx - 1] = 1
            else:
                b[idx] -= T_left

            if j < interior_n - 1:
                A[idx, idx + 1] = 1
            else:
                b[idx] -= T_right

    return A.tocsr(), b

def apply_boundary_conditions(T, T_top, T_bottom, T_left, T_right):
    T[0, :] = T_top
    T[-1, :] = T_bottom
    T[:, 0] = T_left
    T[:, -1] = T_right
    return T

def main():
    n = 160
    T_top = 50
    T_bottom = -50
    T_left = 200
    T_right = 100

    A, b = build_coefficient_matrix(n, T_top, T_bottom, T_left, T_right)
    x = spsolve(A, b)

    T = np.zeros((n + 1, n + 1))
    T = apply_boundary_conditions(T, T_top, T_bottom, T_left, T_right)

    interior_n = n - 1
    T[1:-1, 1:-1] = x.reshape((interior_n, interior_n))

    plt.figure(figsize=(8, 6))
    plt.imshow(T, origin='upper', cmap='hot', extent=[0, 1, 0, 1])
    plt.colorbar(label='Temperature (°C)')
    plt.title('Temperature Distribution in the Plate')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    main()
