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
            
def main():
    imagens, label = pegadados("UDA-TESTE")

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