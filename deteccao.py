import numpy as np
import os
import cv2
import json
import re

def pegadados(caminho = 'UDA-1'):
    imagens = []
    tipo = []
    for (dirname, dirs, files) in os.walk(caminho):
        for filename in files:
            if filename.endswith('.jpg'):
               imagens.append(cv2.imread(dirname+"//"+filename,cv2.IMREAD_ANYCOLOR))
            if filename.endswith('.json'):
                a = open(dirname+"//"+filename).read()
                descricao = json.loads(a)
                tipo.append(descricao['meta']['clinical']['benign_malignant'])
    return imagens, tipo

def divide(img, labels):
    tamanho_teste = int(len(img)//3)

    teste_img = []
    teste_labels = []
    treino_img = []
    treino_labels = []

    print(tamanho_teste)
    for i in range(0,len(img)):
        if i < tamanho_teste:
            teste_img.append(img[i])
            teste_labels.append(labels[i])
        else:
            treino_img.append(img[i])
            treino_labels.append(labels[i])

    return teste_img, treino_img, teste_labels, treino_labels 

def main():
    imagens, label = pegadados("UDA-TESTE")
    teste_img, treino_img, teste_labels, treino_labels = divide(imagens, label)
    print("---teste---\n", teste_labels, "\n---treino---\n", treino_labels)

    for i in label:
        print(i)
    
    for imagen in imagens:
        cv2.imshow("imagen",imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

main()


#cv2.imshow("imagen",imagens[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()