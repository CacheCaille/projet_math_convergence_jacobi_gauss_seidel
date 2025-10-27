import numpy as np
from collections import Counter 
import matplotlib.pyplot as plt
nb_iter = 0;
precision = 0.001
taille = 3

def diagonally_dominant_matrix(n, low=-5, high=5):
    A = np.random.randint(low, high, size=(n, n)).astype(float)
    for i in range(n):
        # Somme des valeurs absolues des autres coefficients
        row_sum = np.sum(np.abs(A[i])) - abs(A[i, i])
        # On rend la diagonale dominante
        A[i, i] = row_sum + np.random.uniform(1, 5)
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

def gauss_seidelv2(A, b, tol=1e-3, max_iter=200):
    n = len(b)
    x = np.zeros((n, 1))
    for iteration in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x, iteration + 1


# Génération d'un dataset de régression (ex : fonction sinusoïdale bruitée)
def generate_data(n=100):
    X = np.linspace(-2*np.pi, 2*np.pi, n).reshape(1, -1)
    Y = np.sin(X) + 0.1 * np.random.randn(1, n)
    return X, Y

# Initialisation du MLP
def init_network(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.1
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.1
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

# Fonctions d'activation
def relu(Z): return np.maximum(0, Z)
def relu_derivative(Z): return (Z > 0).astype(float)

# Forward pass
def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2  # sortie linéaire (pas de sigmoid car régression)
    return Z1, A1, Z2

# Calcul du coût (MSE)
def compute_loss(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# Backpropagation
def backward(X, Y, Z1, A1, Z2, W2):
    m = X.shape[1]
    dZ2 = (Z2 - Y) / m
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T)
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

# Entraînement
def train(X, Y, hidden_size=10, learning_rate=0.01, max_iter=1000, tol=1e-4):
    input_size, output_size = X.shape[0], Y.shape[0]
    W1, b1, W2, b2 = init_network(input_size, hidden_size, output_size)
    losses = []
    for iteration in range(max_iter):
        Z1, A1, Z2 = forward(X, W1, b1, W2, b2)
        loss = compute_loss(Y, Z2)
        losses.append(loss)
        dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, Z2, W2)

        # Mise à jour à la "Gauss-Seidel" (séquentielle)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if iteration > 1 and abs(losses[-2] - losses[-1]) < tol:
            break

    return W1, b1, W2, b2, losses, iteration + 1


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
    #iter_list = []
    #for i in range(5000):
    #    A = diagonally_dominant_matrix(taille)
    #    B = b_matrix(taille)

    #    xk, nb_iter = gauss_seidelv2(A, B)
    #    #print("Matrice x après {0} itérations :\n {1}".format(nb_iter, xk))
    #    iter_list.append(nb_iter)
    #    nb_iter = 0
    #frequences = Counter(iter_list)
    #valeurs = list(frequences.keys())
    #occurrences = list(frequences.values())
    #plt.bar(valeurs, occurrences, color="lightcoral")
    #plt.xlabel("Élément")
    #plt.ylabel("Fréquence")
    #plt.title("Fréquence des éléments dans la liste")
    #plt.show()
    # =============================
# EXPÉRIMENTATION
# =============================

    X, Y = generate_data(100)
    W1, b1, W2, b2, losses, n_iter = train(X, Y, hidden_size=10, learning_rate=0.01, max_iter=2000)

# Prédiction finale
    _, _, Y_pred = forward(X, W1, b1, W2, b2)

# 1️⃣ Courbe d'erreur
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel("Itération")
    plt.ylabel("Erreur quadratique moyenne (MSE)")
    plt.title("Courbe d'erreur de l'apprentissage")
    plt.grid(True)
    plt.show()
    nb_iters = []
    for _ in range(200):
        np.random.seed(None)  # graine aléatoire différente à chaque fois
        _, _, _, _, _, n_iter = train(X, Y, hidden_size=10, learning_rate=0.01, max_iter=2000)
        nb_iters.append(n_iter)


    plt.figure(figsize=(6,4))
    plt.hist(nb_iters, bins=20, color="lightcoral")
    plt.xlabel("Nombre d'itérations jusqu'à convergence")
    plt.ylabel("Fréquence")
    plt.title("Histogramme du nombre d'itérations")
    plt.show()


# 3️⃣ Courbe de prédiction vs réel
    plt.figure(figsize=(6,4))
    plt.plot(X.flatten(), Y.flatten(), label="Valeurs réelles", color="dodgerblue")
    plt.plot(X.flatten(), Y_pred.flatten(), label="Prédictions MLP", color="orange")
    plt.legend()
    plt.title("Prédiction vs Réel (régression)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    
