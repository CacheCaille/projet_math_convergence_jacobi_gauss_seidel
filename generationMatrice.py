import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ==============================================================
#     FONCTIONS GAUSS-SEIDEL
# ==============================================================

precision = 0.001
taille = 3
nb_iter = 0

def diagonally_dominant_matrix(n, low=-5, high=5):
    A = np.random.randint(low, high, size=(n, n)).astype(float)
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 5)
    return A

def b_matrix(n=3):
    return np.random.randint(-10, 10, (n, 1))

def d_matrix(A, n=3):
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = A[i, i]
    return D

def e_matrix(A, n=3):
    E = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i):
            E[i, j] = -A[i, j]
    return E

def f_matrix(A, n=3):
    F = np.zeros((n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            F[i, j] = -A[i, j]
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
            minor = np.delete(np.delete(A, i, axis=0), j, axis=1)
            com[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    return com

def calculate_y(verif_x, b, n=3):
    y = np.zeros((n, 1))
    for i in range(n):
        if verif_x[i, 0] == 0:
            y[i, 0] = 1 if b[i, 0] == 0 else 0
        else:
            y[i, 0] = b[i, 0] / verif_x[i, 0]
    return y

def verify_precision(y):
    for val in y:
        if abs(val - 1) > precision:
            return False
    return True

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


# ==============================================================
#     MLP (RÉSEAU DE NEURONES POUR LA RÉGRESSION)
# ==============================================================

def generate_data(n=100):
    X = np.linspace(-2*np.pi, 2*np.pi, n).reshape(1, -1)
    Y = np.sin(X) + 0.1 * np.random.randn(1, n)
    return X, Y

def init_network(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size) * 0.1
    b1 = np.zeros((hidden_size, 1))
    W2 = np.random.randn(output_size, hidden_size) * 0.1
    b2 = np.zeros((output_size, 1))
    return W1, b1, W2, b2

def relu(Z): return np.maximum(0, Z)
def relu_derivative(Z): return (Z > 0).astype(float)

def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    return Z1, A1, Z2

def compute_loss(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

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

def train(X, Y, hidden_size=10, learning_rate=0.01, max_iter=1000, tol=1e-4):
    input_size, output_size = X.shape[0], Y.shape[0]
    W1, b1, W2, b2 = init_network(input_size, hidden_size, output_size)
    losses = []
    for iteration in range(max_iter):
        Z1, A1, Z2 = forward(X, W1, b1, W2, b2)
        loss = compute_loss(Y, Z2)
        losses.append(loss)

        dW1, db1, dW2, db2 = backward(X, Y, Z1, A1, Z2, W2)
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2

        if iteration > 1 and abs(losses[-2] - losses[-1]) < tol:
            break

    return W1, b1, W2, b2, losses, iteration + 1


# ==============================================================
#     EXPÉRIMENTATION
# ==============================================================

test = False

if test:
    A = diagonally_dominant_matrix(taille)
    B = b_matrix(taille)

    D, E, F = d_matrix(A, taille), e_matrix(A, taille), f_matrix(A, taille)
    d_minus_e = D - E

    inv_d_m_e = np.linalg.inv(d_minus_e)
    inv_d_m_e_f = np.dot(inv_d_m_e, F)
    inv_d_m_e_b = np.dot(inv_d_m_e, B)

    xk = np.dot(inv_d_m_e_f, np.zeros((3, 1))) + inv_d_m_e_b
    print(f"Matrice x après {nb_iter} itérations :\n{xk}")
else:
    X, Y = generate_data(100)
    W1, b1, W2, b2, losses, n_iter = train(X, Y, hidden_size=10, learning_rate=0.01, max_iter=2000)
    _, _, Y_pred = forward(X, W1, b1, W2, b2)

    # Courbe d'erreur
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("Itération")
    plt.ylabel("Erreur quadratique moyenne")
    plt.title("Courbe d'apprentissage du MLP")
    plt.grid(True)
    plt.show()

    # Histogramme du nombre d'itérations jusqu'à convergence
    nb_iters = []
    for _ in range(200):
        _, _, _, _, _, n_iter = train(X, Y, hidden_size=10, learning_rate=0.01, max_iter=2000)
        nb_iters.append(n_iter)

    plt.figure(figsize=(6, 4))
    plt.hist(nb_iters, bins=20, color="lightcoral")
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Fréquence")
    plt.title("Histogramme des itérations jusqu'à convergence")
    plt.show()

    # Prédiction vs valeurs réelles
    plt.figure(figsize=(6, 4))
    plt.plot(X.flatten(), Y.flatten(), label="Valeurs réelles", color="dodgerblue")
    plt.plot(X.flatten(), Y_pred.flatten(), label="Prédictions MLP", color="orange")
    plt.legend()
    plt.title("Prédiction vs Réel (MLP)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()
