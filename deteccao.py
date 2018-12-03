import numpy as np
import os
import cv2
import json
import re

from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation

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

def CNN(largura, altura, canais, classes):
    tamanho_entrada = (altura, largura, canais)
 
    modelo = Sequential()
    modelo.add(Conv2D(6, (5, 5), padding="same", input_shape=tamanho_entrada))
    modelo.add(Activation("relu"))
    modelo.add(MaxPooling2D((2,2)))
    modelo.add(Conv2D(16, (5, 5)))
    modelo.add(Activation("relu"))
    modelo.add(MaxPooling2D((2,2)))
    modelo.add(Flatten())
    modelo.add(Dense(120))
    modelo.add(Activation("relu"))
    modelo.add(Dense(84))
    modelo.add(Activation("relu"))
    modelo.add(Dense(classes))
    modelo.add(Activation("softmax"))
 
    return modelo

def divide(img, labels):
    tamanho_teste = int(len(img)//3)

    teste_img = []
    teste_labels = []
    treino_img = []
    treino_labels = []

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
    largura, altura, canais = imagens[0].shape
    cnn = CNN(largura,altura,canais,2)
    """
    for imagen in imagens:
        cv2.imshow("imagen",imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """

main()


#cv2.imshow("imagen",imagens[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()