# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import os


def top_5_accuracy(y_true, y_pred_probs):
    top_5_preds = np.argsort(y_pred_probs, axis=1)[:, -5:]
    correct_top_5 = sum(y_true[i] in top_5_preds[i] for i in range(len(y_true)))
    return correct_top_5 / len(y_true)


def load_data(dir_test_path, img_size=(224, 224)):
    datagen = ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        dir_test_path,
        target_size=img_size,
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )
    return test_gen


def evaluate_model(model_path, dir_test_path, img_size=(224, 224)):
    model = load_model(model_path)
    test_gen = load_data(dir_test_path, img_size)
    y_true = test_gen.classes
    class_indices = test_gen.class_indices
    class_labels = list(class_indices.keys())

    # probabilidades para todas as classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Top-5 Accuracy
    top_5_acc = top_5_accuracy(y_true, y_pred_probs)

    # métricas
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"Model Name: {model_name}")
    print(f"Top-5 Accuracy: {top_5_acc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Calcular AUC-ROC e salvar a curva
    y_true_binary = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
    fpr, tpr, roc_auc = {}, {}, {}

    for i, label in enumerate(class_labels):
        fpr[label], tpr[label], _ = roc_curve(y_true_binary[:, i], y_pred_probs[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    # Curva ROC macro
    plt.figure(figsize=(10, 8))
    for label in class_labels:
        plt.plot(fpr[label], tpr[label], label=f'{label} (AUC = {roc_auc[label]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.5)
    plt.savefig('roc_curve.png')
    plt.show()


dir_test_path = '/home/patriciiarodrigs/TCC/data_img/test'
model_path = '/home/patriciiarodrigs/TCC/projeto/modelo_xception_40_epocas/xception_model_exception.h5'


evaluate_model(model_path, dir_test_path)

