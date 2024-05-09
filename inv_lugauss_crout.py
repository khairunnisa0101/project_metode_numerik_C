import numpy as np
from numpy.linalg import inv

def inverse_matrix(A, b):
    
    #Solusi sistem persamaan linier menggunakan metode invers matriks.

    A_inv = inv(A)
    x = np.dot(A_inv, b)
    return x

def gauss_elimination(A, b):
    
    #Solusi sistem persamaan linier menggunakan metode eliminasi Gauss.
    
    n = len(b)
    
    # Eliminasi maju
    for i in range(n):
        for j in range(i+1, n):
            ratio = A[j,i]/A[i,i]
            for k in range(n):
                A[j,k] = A[j,k] - ratio * A[i,k]
            b[j] = b[j] - ratio * b[i]
    
    # Substitusi mundur
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i,i+1:], x[i+1:])) / A[i,i]
    
    return x

def crout_decomposition(A):
    
    #Dekomposisi matriks menggunakan metode Crout.

    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    
    for i in range(n):
        U[i,i] = 1
        
    for i in range(n):
        for j in range(i, n):
            sum = 0
            for k in range(i):
                sum += L[i,k] * U[k,j]
            U[i,j] = A[i,j] - sum
            
        for j in range(i, n):
            if i == j:
                L[i,i] = 1
            else:
                sum = 0
                for k in range(i):
                    sum += L[j,k] * U[k,i]
                L[j,i] = (A[j,i] - sum) / U[i,i]
    
    return L, U

def crout_solve(L, U, b):
    
    #Solusi sistem persamaan linier menggunakan dekomposisi Crout.
    
    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)
    
    # Solusi Ly = b
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]
    
    # Solusi Ux = y
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i+1:], x[i+1:])) / U[i,i]
    
    return x

# Uji coba
A = np.array([[4, -2, 1],
              [1, 3, -1],
              [2, -1, 5]])

b = np.array([10, -1, 3])

# Solusi menggunakan metode invers matriks
x_inv = inverse_matrix(A, b)
print("Solusi menggunakan metode invers matriks:")
print(x_inv)

# Solusi menggunakan metode eliminasi Gauss
x_gauss = gauss_elimination(A.copy(), b.copy())
print("\nSolusi menggunakan metode eliminasi Gauss:")
print(x_gauss)

# Solusi menggunakan dekomposisi Crout
L, U = crout_decomposition(A.copy())
x_crout = crout_solve(L, U, b.copy())
print("\nSolusi menggunakan dekomposisi Crout:")
print(x_crout)
