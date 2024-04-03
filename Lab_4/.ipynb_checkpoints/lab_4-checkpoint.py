import numpy as np
import matplotlib.pyplot as plt

# Дані
x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

# Розділення даних на класи
x_0 = x_train[y_train == 1]
x_1 = x_train[y_train == -1]

# Параметри моделі для кожного класу
mean_0 = np.mean(x_0, axis=0)
cov_0 = np.cov(x_0.T)
mean_1 = np.mean(x_1, axis=0)
cov_1 = np.cov(x_1.T)

# Функція щільності гаусового розподілу
def gaussian_density(x, mean, cov):
    d = len(mean)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    norm_const = 1.0 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
    exp_term = np.exp(-0.5 * np.dot(np.dot((x - mean).T, inv), x - mean))
    return norm_const * exp_term

# Функція класифікації
def predict(x):
    p_0 = gaussian_density(x, mean_0, cov_0)
    p_1 = gaussian_density(x, mean_1, cov_1)
    if p_1 > p_0:
        return 1

    else:
        return -1

# Графік
x_min, x_max = 0, 50
y_min, y_max = 0, 80
step = 0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
Z = np.array([predict(np.array([xx, yy]).reshape(2, 1)) for xx, yy in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=100, edgecolors='k')
plt.title('Наївний гаусовський байєсівський класифікатор')
plt.xlabel('Довжина')
plt.ylabel('Ширина')
plt.show()
