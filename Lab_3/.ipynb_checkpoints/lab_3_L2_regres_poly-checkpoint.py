# Приклад роботи регуляризатора L2

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10.1, 0.1)
y = np.array([a ** 3 - 10 * a ** 2 + 3 * a + 500 for a in x])  # функція у вигляді полінома x^3 - 10x^2 + 3x + 500
x_train, y_train = x[::2], y[::2]
N = 13  # розмір простору ознак (степінь полінома N-1)
L = 20  # при збільшенні N збільшується L (кратно): 12; 0.2   13; 20    15; 5000

X = np.array([[a ** n for n in range(N)] for a in x])  # матриця вхідних векторів
IL = np.array([[L if i == j else 0 for j in range(N)] for i in range(N)])  # матриця lambda*I
IL[0][0] = 0  # перший коефіцієнт не регуляризується
X_train = X[::2]  # навчальна вибірка
Y = y_train  # навчальна вибірка

# обчислення коефіцієнтів за формулою w = (XT*X + lambda*I)^-1 * XT * Y
A = np.linalg.inv(X_train.T @ X_train + IL) # Тут і нижче @ - оператор точного матричного множення
w = Y @ X_train @ A
print(w)

# відображення початкового графіка та прогнозу
yy = [np.dot(w, x) for x in X]
plt.plot(x, yy) # прогноз моделі
plt.plot(x, y) # справжня поведідка функції
plt.grid(True)
plt.show()