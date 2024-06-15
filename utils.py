import os
import cv2
from lbp import extract_lbp_features
import numpy as np
import matplotlib.pyplot as plt

def load_images_from_folder(folder, max_images_per_class=None):
    images = []
    labels = []
    for class_label in ['covid', 'normal']:
        class_folder = os.path.join(folder, class_label)
        count = 0
        print(f"Carregando imagens da classe: {class_label}")
        for filename in os.listdir(class_folder):
            if max_images_per_class and count >= max_images_per_class:
                break
            img_path = os.path.join(class_folder, filename)
            print(f"Lendo imagem: {img_path}")
            image = cv2.imread(img_path)
            if image is not None:
                print(f"Imagem lida com sucesso: {img_path}")
                features = extract_lbp_features(image)
                images.append(features)
                labels.append(class_label)
                count += 1
            else:
                print(f"Erro ao ler a imagem: {img_path}")
    print("Todas as imagens foram carregadas")
    return np.array(images), np.array(labels)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão sem normalização')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Classe real')
    plt.xlabel('Classe predita')
    plt.tight_layout()
