import matplotlib.pyplot as plt
import os
import numpy as np

# Задайте шлях до вашої теки з зображеннями
image_dir = "images"

# Список файлів у вказаній директорії
image_files = os.listdir(image_dir)

# Виберіть перші n зображень зі списку
n_samples = 8
selected_images = image_files[:n_samples]

# Виведення зображень
plt.figure(figsize=(16, 8))
rows, cols = 2, 4
for i, image_file in enumerate(selected_images):
    plt.subplot(rows, cols, i + 1)
    # Зчитуємо та відображаємо зображення
    image_path = os.path.join(image_dir, image_file)
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.xticks(())
    plt.yticks(())

plt.show()


from PIL import Image
images = []
image_shape = (64, 64)
# Зміна розміру та конвертація кольору зображення перед додаванням до списку
for image_file in selected_images:
    image_path = os.path.join(image_dir, image_file)
    image = Image.open(image_path).convert('L')  # Конвертація в чорно-біле
    image = image.resize(image_shape)  # Зміна розміру
    images.append(np.array(image).flatten())
# Виведення перших кількох зображень
plt.figure(figsize=(10, 5))
for i in range(3):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].reshape(image_shape), cmap='gray')
    plt.axis('off')  # Вимкнути відображення осей
    plt.title(f"Image {i+1}")

plt.show()

from sklearn.decomposition import PCA

base_size = image_shape[0] * image_shape[1]


def compress_and_show(compress_ratio):
    n_samples = 3  # Кількість зображень для відображення
    selected_images = image_files[:n_samples]
    global images
    images = np.array(images)

    # Створення моделі PCA та стискання зображень
    base_size = images.shape[1]
    model_pca = PCA(n_components=int(base_size * compress_ratio))
    model_pca.fit(images)
    images_compressed = model_pca.transform(images)

    # Відновлення зображень
    images_restored = model_pca.inverse_transform(images_compressed)
    images_restored = images_restored.reshape((-1, *image_shape))  # Повернення у формування зображень

    # Відображення стиснутих та відновлених зображень
    plt.figure(figsize=(16, 8))
    rows, cols = 2, 4
    for i in range(n_samples):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images_restored[i], interpolation='none', cmap='gray')
        plt.xticks(())
        plt.yticks(())

    plt.show()


compress_and_show(0.0005)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Ваш варіант даних
x_train_2 = np.array([[34, 11], [34, 38], [40, 8], [42, 18], [33, 41], [39, 47], [11, 40], [21, 44], [19, 28], [49, 20]])
y_train_2 = np.array([-1, -1, -1, 1, 1, 1, -1, 1, -1, 1])

# Тестовий набір даних
x_test_2 = np.array([[35, 10], [36, 39], [41, 9], [43, 19], [32, 42], [38, 48], [12, 39], [22, 45], [20, 29], [48, 21]])
y_test_2 = np.array([-1, -1, -1, 1, 1, 1, -1, 1, -1, 1])

# Визначення класифікатора та навчання моделі
k = 3  # Кількість найближчих сусідів
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train_2, y_train_2)

# Прогнозування класів для тестових даних
y_pred_2 = knn.predict(x_test_2)

# Оцінка точності класифікації
accuracy_2 = accuracy_score(y_test_2, y_pred_2)
print(f"Точність класифікації методом k найближчих сусідів: {accuracy_2:.2f}")

# Вивід таблиці частот точності класифікації
print("Таблиця частот точності класифікації:")
print("--------------------------------------------------")
print("| Клас | Правильно класифіковано | Неправильно класифіковано |")
print("--------------------------------------------------")
for target in np.unique(y_test_2):
    correct = np.sum((y_test_2 == target) & (y_pred_2 == target))
    incorrect = np.sum((y_test_2 == target) & (y_pred_2 != target))
    print(f"|  {target}  | {correct:^25} | {incorrect:^28} |")
print("--------------------------------------------------")

# Візуалізація
plt.figure(figsize=(10, 6))
plt.scatter(x_test_2[:, 0], x_test_2[:, 1], c=y_pred_2, cmap='viridis', s=50)
plt.title('Класифікація методом k найближчих сусідів')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.colorbar(label='Клас')
plt.show()
