0	# -*- coding: utf-8 -*-
"""
bird_classifier_image.py
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Sequential
import pathlib
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score



# Diretórios de dados
data_dir_train = 'TCC/data_img/train'
data_dir_test = 'TCC/data_img/test'
data_dir_valid = 'TCC/data_img/valid'

# Parâmetros
batch_size = 32
img_height = 224
img_width = 224


AUTOTUNE = tf.data.AUTOTUNE

# Carregando datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_valid,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Nome das classes
class_names = train_ds.class_names
print("Classes:", class_names)
num_classes = len(class_names)


# Normalização e prefetch
AUTOTUNE = tf.data.AUTOTUNE
#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalização manual
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Aumento de Dados
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Aplicando aumento de dados e salvando exemplos aumentados
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.savefig('data_augmentation_examples.png')  # Salva a imagem
plt.close()

# Dropout
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, name="outputs")
])

# Compilando o modelo com dropout
model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])



# Construindo o modelo
#num_classes = len(class_names)
#model = Sequential([
#    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#    layers.Conv2D(16, 3, padding='same', activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(32, 3, padding='same', activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(64, 3, padding='same', activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Flatten(),
#    layers.Dense(128, activation='relu'),
#    layers.Dense(num_classes, name="outputs")
#])

# Compilando o modelo
#model.compile(
#    optimizer='adam',
#    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#    metrics=['accuracy']
#)



model.summary()

# Treinando o modelo
epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Salvando gráfico de resultados de treino
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig("training_validation_results.png")
plt.close()

# Predições em novos dados
path_test_images = []
for subdir in os.listdir(data_dir_test):
    subdir_path = os.path.join(data_dir_test, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                path_test_images.append(os.path.join(subdir_path, file))

num_images = len(path_test_images)
num_subplots = min(num_images, 6)

# Visualizando predições
num_rows = min(num_subplots, 2)
num_cols = (num_subplots + 1) // 2  # Arredonda para cima

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

for ind, p in enumerate(path_test_images[:num_subplots]):
    img = tf.keras.utils.load_img(p, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Criar batch com 1 imagem
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    ax = axes.flat[ind]
    ax.imshow(img)
    ax.set_title(f"Predicted: {predicted_class}")
    ax.axis("off")

plt.savefig("test_predictions.png")
plt.close()


# Predição sobre novos dados
def load_and_preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Cria um lote de tamanho 1
    return img_array


# Predições e métricas
y_true = []
y_pred = []
y_pred_proba = []

for class_folder in os.listdir(data_dir_test):
    class_path = os.path.join(data_dir_test, class_folder)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            if image_name.endswith('.jpg') or image_name.endswith('.png'):
                image_path = os.path.join(class_path, image_name)
                img_array = load_and_preprocess_image(image_path)
                predictions = model.predict(img_array)
                probas = tf.nn.softmax(predictions[0]).numpy()

                y_true.append(class_names.index(class_folder))
                y_pred.append(np.argmax(probas))
                y_pred_proba.append(probas)

# Convertendo para numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_proba = np.array(y_pred_proba)

# Calculando métricas
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

# Exibindo métricas
print("Acurácia: {:.4f}".format(accuracy))
print("Precisão (Weighted): {:.4f}".format(precision))
print("Revocação (Weighted): {:.4f}".format(recall))
print("F1 Score (Weighted): {:.4f}".format(f1))
print("AUC ROC (Weighted): {:.4f}".format(auc_roc))


output_dir = 'TCC/projeto/'
os.makedirs(output_dir, exist_ok=True)
model.save(os.path.join(output_dir, '30_model.h5'))
