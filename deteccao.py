#importações extras
import pickle
import numpy as np
import os
import cv2
import json
import re
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
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
    
    print("testY  normal: ",testY)

    #converte os labels para binarios
    testY = to_categorical(testY, 2)
    print("testY: ",testY)
    trainY = to_categorical(trainY, 2)

    largura, altura, canais = imagens[0].shape
    
    print("[INFO] inicializando e otimizando a CNN...")
    cnn = CNN(altura, largura, canais, 2)
    cnn.compile(optimizer=SGD(0.01), loss="categorical_crossentropy", metrics=["accuracy"])
    print("[INFO] treinando a CNN...")
    H = cnn.fit(trainX, trainY, batch_size=128, epochs=1, verbose=2, validation_data=(testX, testY))

    print("[INFO] avaliando a CNN...")
    predictions = cnn.predict(testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(label) for label in range(10)]))
    
    s = pickle.dumps(cnn)
    cnn = pickle.loads(s)
    print(cnn.predict(imagens))
    print(label)


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