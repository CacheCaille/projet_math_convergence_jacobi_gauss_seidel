import numpy as np


def diagonally_dominant_matrix(n, low=-5, high=5):
    A = np.random.randint(low, high, size=(n, n))
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + np.random.rand()  
    return A

def b_matrix(n = 3):
    return np.random.randint(-10,10,(n,1))
    
def d_matrix(matrix_A, n = 3):
    D  = np.zeros((n,n))
    for i in range(n):
        D[i,i] = A[i,i]
    return D
    
def e_matrix(matrix_A, n = 3):
    E = np.zeros((n,n))
    for i in range(n):
        if i == 0:
            continue
        for j in range(i):
            E[i,j] = -A[i,j]
    return E
    
def f_matrix(matrix_A, n = 3):
    
    F = np.zeros((n,n))
    for i in range(n-1):
        for j in range(n):
            if i < j:
                F[i,j] = -A[i,j]
    return F


def determinant(matrix):
    if matrix.shape == (1, 1):
        return matrix[0, 0]
    if matrix.shape == (2, 2):
        return matrix[0,0]*matrix[1,1] - matrix[0,1]*matrix[1,0]
    
    det = 0
    for c in range(matrix.shape[1]):
        minor = np.delete(np.delete(matrix, 0, axis=0), c, axis=1)
        det += ((-1)**c) * matrix[0, c] * determinant(minor)
    return det

def comatrix(A):
    n = A.shape[0]
    com = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Sous-matrice sans la ligne i et la colonne j
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            # Cofacteur
            com[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    
    return com

def gauss_sedeil(matrix_D_E_F, matrix_D_E_B):
    xk = np.zeros((3,1))
    xk1 = np.dot(matrix_D_E_F,xk) + matrix_D_E_B
    
    for i in range(10):
        xk1 = np.dot(matrix_D_E_F,xk1) + matrix_D_E_B
    
    return xk1

taille = 3
A = diagonally_dominant_matrix(taille)
B = b_matrix(taille)


print("Matrice A :\n", A)
print("Matrice b :\n", B)

D = d_matrix(A,taille)
E = e_matrix(A,taille)
F = f_matrix(A,taille)

print("Matrice diagonale :\n", D)
print("Matrice inférieure :\n", E)
print("Matrice supérieure :\n", F)

d_minus_e = D - E

print("Matrice D - E :\n",d_minus_e)

det_d_m_e = determinant(d_minus_e)

print("Déterminant de D - E :\n", det_d_m_e)

com = comatrix(d_minus_e)

print("Commatrice de D - E :\n", com)

inv_d_m_e = np.dot(1/det_d_m_e,com.T)

print("Matrice inverse de D - E :\n", inv_d_m_e)

inv_d_m_e_f = np.dot(inv_d_m_e,F)

print("Matrice inverse de D - E * F :\n",inv_d_m_e_f)

inv_d_m_e_b = np.dot(inv_d_m_e,B)

print("Matrice inverse de D - E * b :\n", inv_d_m_e_b)

xk = gauss_sedeil(inv_d_m_e_f,inv_d_m_e_b)
print("Matrice x après 10 itérations :\n", xk)

verif_x = np.dot(A,xk)
print("Vérification de x :\n", verif_x)
print("Matrice b (rappel) :\n", B)

y = verif_x / B

print("Coefficient d'approchement y :\n",y)
