import numpy as np
from collections import Counter 
import matplotlib.pyplot as plt
nb_iter = 0;
precision = 0.001
taille = 3

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

def calculate_y(verif_x,b,n = 3):
    y = np.zeros((3,1))
    for i in range(n):
        if verif_x[i,0] == 0 :
            if b[i,0] == 0 :
                y[i,0] = 1
            else :
                y[i,0] = 0
        else :
            y[i,0] = b[i,0]/verif_x[i,0]
    return y
        
def verify_precision(y):
    is_precision_ok = True
    global precision;
    global taille;
    for i in range(taille):
        if abs(y[i,0] - 1) > precision:
            is_precision_ok = False
    return is_precision_ok
        
def gauss_sedeil(matrix_D_E_F, matrix_D_E_B, matrix_A, matrix_B):
    xk = np.zeros((3,1))
    xk1 = np.dot(matrix_D_E_F,xk) + matrix_D_E_B
    global nb_iter;
    global taille;
    for i in range(200):
        nb_iter = nb_iter + 1
        xk1 = np.dot(matrix_D_E_F,xk1) + matrix_D_E_B
        verif_x = np.dot(A,xk1)
        y = calculate_y(verif_x,B, taille)
        is_precision_ok = verify_precision(y)
        if is_precision_ok:
            break;
    return xk1


test = False;        
if test:
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

    xk = gauss_sedeil(inv_d_m_e_f,inv_d_m_e_b, A, B)
    print("Matrice x après {0} itérations :\n {1}".format(nb_iter, xk))

    verif_x = np.dot(A,xk)
    print("Vérification de x :\n", verif_x)
    print("Matrice b (rappel) :\n", B)

    y = calculate_y(verif_x, B, taille)

    print("Coefficient d'approchement y :\n",y)
else:
    iter_list = []
    for i in range(5000):
        A = diagonally_dominant_matrix(taille)
        B = b_matrix(taille)

        D = d_matrix(A,taille)
        E = e_matrix(A,taille)
        F = f_matrix(A,taille)
        d_minus_e = D - E
        det_d_m_e = determinant(d_minus_e)
        com = comatrix(d_minus_e)
        inv_d_m_e = np.dot(1/det_d_m_e,com.T)
        inv_d_m_e_f = np.dot(inv_d_m_e,F)
        inv_d_m_e_b = np.dot(inv_d_m_e,B)
        xk = gauss_sedeil(inv_d_m_e_f,inv_d_m_e_b, A, B)
        if nb_iter == 200:
            print(A)
        #print("Matrice x après {0} itérations :\n {1}".format(nb_iter, xk))
        iter_list.append(nb_iter)
        nb_iter = 0
    frequences = Counter(iter_list)
    valeurs = list(frequences.keys())
    occurrences = list(frequences.values())
    plt.bar(valeurs, occurrences, color="lightcoral")
    plt.xlabel("Élément")
    plt.ylabel("Fréquence")
    plt.title("Fréquence des éléments dans la liste")
    plt.show()

    
