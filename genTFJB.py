import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# ===========================
# Paramètres généraux
# ===========================
taille = 3
precision = 0.001
nb_samples = 500

# ===========================
# Fonctions pour générer les systèmes
# ===========================
def diagonally_dominant_matrix(n, low=-5, high=5):
    A = np.random.randint(low, high, size=(n, n)).astype(float)
    for i in range(n):
        row_sum = np.sum(np.abs(A[i])) - abs(A[i, i])
        A[i, i] = row_sum + np.random.uniform(1, 5)
    return A

def b_matrix(n=3):
    return np.random.randint(-10, 10, (n, 1))

# ===========================
# Méthode de Jacobi
# ===========================
def jacobi(A, b, tol=1e-3, max_iter=200):
    n = len(b)
    x = np.zeros((n, 1))
    for iteration in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = np.dot(A[i, :], x) - A[i, i]*x[i]
            x_new[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x, iteration + 1

# ===========================
# Génération du dataset
# ===========================
def generate_dataset(nb_samples=500, taille=3):
    X = []
    Y = []
    it_counts = []
    for _ in range(nb_samples):
        A = diagonally_dominant_matrix(taille)
        b = b_matrix(taille)
        x_sol, it = jacobi(A, b)
        input_vector = np.concatenate([A.flatten(), b.flatten()])
        X.append(input_vector)
        Y.append(x_sol.flatten())
        it_counts.append(it)
    return np.array(X), np.array(Y), np.array(it_counts)

X, Y, iterations = generate_dataset(nb_samples=nb_samples, taille=taille)

# ===========================
# Split train/test
# ===========================
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# ===========================
# Construction du modèle TensorFlow
# ===========================
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(Y_train.shape[1])  # sortie = 3 valeurs (vecteur solution)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='mse',
              metrics=['mae'])

# ===========================
# Entraînement du modèle
# ===========================
history = model.fit(
    X_train, Y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ===========================
# Courbe d'apprentissage
# ===========================
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel("Époque")
plt.ylabel("MSE")
plt.title("Courbe d'apprentissage du MLP (TensorFlow, Jacobi)")
plt.legend()
plt.show()

# ===========================
# Histogramme du nombre d'itérations Jacobi
# ===========================
plt.figure()
plt.hist(iterations, bins=20, color='lightgreen', edgecolor='black')
plt.xlabel("Nombre d'itérations Jacobi")
plt.ylabel("Fréquence")
plt.title("Histogramme des itérations Jacobi")
plt.show()

# ===========================
# Comparaison prédiction vs réel
# ===========================
Y_pred = model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(Y_test.flatten(), Y_pred.flatten(), alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--', label='y=x')
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions du modèle")
plt.title("Comparaison prédictions vs réel (TensorFlow, Jacobi)")
plt.legend()
plt.show()
