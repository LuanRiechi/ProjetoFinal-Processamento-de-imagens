import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from lbp import extract_lbp_features
from utils import load_images_from_folder, plot_confusion_matrix
import matplotlib.pyplot as plt
import time

def main():
    start_time = time.time()

    # Carregar dataset
    folder = 'dataset'
    print("Carregando dataset...")
    images, labels = load_images_from_folder(folder, max_images_per_class=5)  # Ajuste este número para testar com menos imagens
    print("Dataset carregado com sucesso")

    # Dividir dados em treinamento e teste
    print("Dividindo dados em treinamento e teste...")
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)
    print("Divisão concluída")

    # Treinar classificador SVM
    print("Treinando classificador SVM...")
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    print("Classificador treinado com sucesso")

    # Prever no conjunto de teste
    print("Realizando previsões no conjunto de teste...")
    y_pred = clf.predict(X_test)
    print("Previsões concluídas")

    # Avaliar modelo
    print("Avaliando o modelo...")
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Imprimir e salvar os resultados
    print("Acurácia:", accuracy)
    print("Matriz de Confusão:")
    print(cm)

    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)

    # Salvar acurácia
    with open(os.path.join(results_folder, 'accuracy.txt'), 'w') as f:
        f.write(f"Acurácia: {accuracy}\n")

    # Salvar matriz de confusão
    np.savetxt(os.path.join(results_folder, 'confusion_matrix.txt'), cm, fmt='%d')

    # Plotar e salvar a matriz de confusão
    plot_confusion_matrix(cm, classes=['covid', 'normal'])
    plt.savefig(os.path.join(results_folder, 'confusion_matrix.png'))
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tempo de execução: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    main()
