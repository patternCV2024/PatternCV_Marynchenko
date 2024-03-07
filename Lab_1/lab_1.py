
import numpy as np
import matplotlib.pyplot as plt

def classifier(x_train, y_train):

    change = True

    n_train = len(x_train)  # розмір навчальної вибірки
    w = [0, -1]  # початкове значення вектора w
    a = lambda x: np.sign(x[0] * w[0] + x[1] * w[1])  # правило класифікації
    L = 0.1  # крок зміни ваги
    e = 0.1  # невелике додаток до w0, щоб забезпечити зазор між лінією розділення та областю
    count = 0
    last_error_index = -1  # індекс останньої помилкової спостереження
    while change and count<100:
        change = False
        for i in range(n_train):  # ітерація по спостереженням
            if y_train[i] * a(x_train[i]) < 0:  # якщо помилка класифікації,
                w[0] = w[0] + L * y_train[i]  # то коригування ваги w0
                last_error_index = i
                change = True


        Q = sum([1 for i in range(n_train) if y_train[i] * a(x_train[i]) < 0])
        if Q == 0:  # показник якості класифікації (кількість помилок)
            break  # зупинка, якщо всі класифікуються правильно
        count += 1
    if last_error_index > -1:
        w[0] = w[0] + e * y_train[last_error_index]

    print(w)

    line_x = list(range(max(x_train[:, 0])))  # створення графіка роздільної лінії
    line_y = [w[0] * x for x in line_x]

    x_0 = x_train[y_train == 1]  # формування точок для 1-го
    x_1 = x_train[y_train == -1]  # і 2-го класів

    plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
    plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
    plt.plot(line_x, line_y, color='green')

    plt.xlim([0, 45])
    plt.ylim([0, 75])
    plt.ylabel("довжина")
    plt.xlabel("ширина")
    plt.grid(True)
    plt.show()
x_train_1 = np.array([[10, 44], [38, 30], [43, 37], [50, 27], [47, 10], [ 8, 36], [25, 21], [45,  9], [42, 42], [29, 36]])
y_train_1 = np.array([-1,  1,  -1, 1, -1,  1,  1, 1,  -1, -1])
classifier(x_train_1, y_train_1)