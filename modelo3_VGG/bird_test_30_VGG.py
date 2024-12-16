# -*- coding: utf-8 -*-
"""
bird_classifier_image.py
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score

data_dir_train = '/home/patriciiarodrigs/TCC/data_img/train'
data_dir_test = '/home/patriciiarodrigs/TCC/data_img/test'
data_dir_valid = '/home/patriciiarodrigs/TCC/data_img/valid'

# Parâmetros
batch_size = 32
img_height = 224
img_width = 224

# Geradores de dados para carregamento e aumento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    data_dir_train,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

val_ds = test_datagen.flow_from_directory(
    data_dir_valid,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

# Nome das classes
class_names = list(train_ds.class_indices.keys())
print("Classes:", class_names)
num_classes = len(class_names)

#  modelo VGG16
base_model = VGG16(
    include_top=False,  # Não inclui as camadas de classificação
    weights='imagenet',  # Usando pesos do ImageNet
    input_shape=(img_height, img_width, 3)  # Define o formato de entrada
)


base_model.trainable = False

# Construção do modelo
model = models.Sequential([
    layers.InputLayer(input_shape=(img_height, img_width, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# Treinando o modelo VGG16
epochs = 30
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


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


plt.savefig("training_validation_results_vgg.png")
plt.close()

# Predição em novos dados
path_test_images = []
for subdir in os.listdir(data_dir_test):
    subdir_path = os.path.join(data_dir_test, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.endswith(".jpg") or file.endswith(".png"):
                path_test_images.append(os.path.join(subdir_path, file))

num_images = len(path_test_images)
num_subplots = min(num_images, 6)


num_rows = min(num_subplots, 2)
num_cols = (num_subplots + 1) // 2

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

for ind, p in enumerate(path_test_images[:num_subplots]):
    img = tf.keras.utils.load_img(p, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    ax = axes.flat[ind]
    ax.imshow(img)
    ax.set_title(f"Predicted: {predicted_class}")
    ax.axis("off")


plt.savefig("test_predictions_vgg.png")
plt.close()


output_dir = '/home/patriciiarodrigs/TCC/projeto/'
os.makedirs(output_dir, exist_ok=True)
model.save(os.path.join(output_dir, 'vgg_model.h5'))

print("Modelo salvo com sucesso.")
