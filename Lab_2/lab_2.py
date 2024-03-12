import numpy as np
import matplotlib.pyplot as plt
import openpyxl

wb = openpyxl.load_workbook('data.xlsx')
ws = wb.active

x_train_0 = [ws.cell(row=i,column=1).value for i in range(1,ws.max_row+1)]
x_train_1 = [ws.cell(row=i,column=2).value for i in range(1,ws.max_row+1)]
x_train = []
x_train_item = []
for i in range(len(x_train_0)):
    x_train_item.append(x_train_0[i])
    x_train_item.append(x_train_1[i])
    x_train.append(x_train_item)
    x_train_item = []

y_train = [ws.cell(row=i,column=3).value for i in range(1,ws.max_row+1)]

#x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = [x + [1] for x in x_train]  # Додаємо зміщення для кожного прикладу
x_train = np.array(x_train)
y_train = np.array(y_train)

pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)  # Обчислення підсумку
xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)  # Обчислення підсумку зовнішнього добутку
w = np.dot(pt, np.linalg.inv(xxt))  # Обчислення вагових коефіцієнтів
print(w)

line_x = list(range(max(x_train[:, 0])))    # формування координат для лінії розділення
line_y = [-x*w[0]/w[1] - w[2]/w[1] for x in line_x]

x_0 = x_train[y_train == 1]                 # формування точок для класу 1
x_1 = x_train[y_train == -1]                # і класу -1

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')  # відображення точок класу 1 червоним кольором
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')  # відображення точок класу -1 синім кольором
plt.plot(line_x, line_y, color='green')  # відображення лінії розділення зеленим кольором

plt.xlim([0, 60])
plt.ylim([0, 51])
plt.ylabel("довжина")
plt.xlabel("ширина")
plt.grid(True)
plt.show()