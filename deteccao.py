#importações extras
import numpy as np
import os
import cv2
import json
import re
from sklearn.model_selection import train_test_split
#importações para a CNN
from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation

from keras.optimizers import SGD
from keras.utils import to_categorical


def pegadados(caminho = 'UDA-1'):
    imagens = np.array([])
    tipo = np.array([])
    for (dirname, dirs, files) in os.walk(caminho):
        for filename in files:
            if filename.endswith('.jpg'):
                if imagens.shape[0] != 0:
                    imagens = np.append(imagens, [cv2.imread(dirname+"//"+filename,cv2.IMREAD_ANYCOLOR)])
                else:
                    imagens = np.array([cv2.imread(dirname+"//"+filename,cv2.IMREAD_ANYCOLOR)])
            if filename.endswith('.json'):
                a = open(dirname+"//"+filename).read()
                descricao = json.loads(a)
                if descricao['meta']['clinical']['benign_malignant'] == "benign":
                    tipo = np.append(tipo, 0) #cancer benigno
                else:
                    tipo = np.append(tipo, 1) #cancer maligno
                    
    print(imagens.shape)
    imagens = imagens.reshape(tipo.shape[0],767,1022,3)
    print(imagens.shape)
    return imagens, tipo

def CNN(altura, largura, canais, classes):
    tamanho_entrada = (largura, altura, canais)
 
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

def main():
    imagens, label = pegadados("UDA-TESTE")

    (trainX, testX, trainY, testY) = train_test_split(imagens, label)#dividir teste e treino
    
    #converte os labels para binarios
    testY = to_categorical(testY, 2)
    trainY = to_categorical(trainY, 2)

    largura, altura, canais = imagens[0].shape
    
    cnn = CNN(altura, largura, canais, 2)
    cnn.compile(optimizer=SGD(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    H = cnn.fit(trainX, trainY, batch_size=128, epochs=20, verbose=2, validation_data=(testX, testY))

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