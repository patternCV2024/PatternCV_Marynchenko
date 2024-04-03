import numpy as np

# Дані
x_train = np.array([[47, 13], [25, 11], [19, 17], [7, 16], [32, 12], [46, 26], [48, 24], [20, 14], [34, 43], [43, 48]])
y_train = np.array([ 1, -1,  1,  1,  1,  1,  1, -1,  1,  1])

mw1, ml1 = np.mean(x_train[y_train == 1], axis=0)
mw_1, ml_1 = np.mean(x_train[y_train == -1], axis=0)

# формула для обчислення дисперсії тут трохи інша 1/N*sum(...)
sw1, sl1 = np.var(x_train[y_train == 1], axis=0)
sw_1, sl_1 = np.var(x_train[y_train == -1], axis=0)

print('Середнє: ', mw1, ml1, mw_1, ml_1)
print('Дисперсії:', sw1, sl1, sw_1, sl_1)

x = [40, 10]  # довжина, ширина жука

a_1 = lambda x: -(x[0] - ml_1) ** 2 / (2 * sl_1) - (x[1] - mw_1) ** 2 / (2 * sw_1) # Перший класифікатор
a1 = lambda x: -(x[0] - ml1) ** 2 / (2 * sl1) - (x[1] - mw1) ** 2 / (2 * sw1) # Другий класифікатор
y = np.argmax([a_1(x), a1(x)]) # Обираємо максимум

print('Номер класу (0 - гусениця, 1 - божа корівка): ', y)
