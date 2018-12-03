import numpy as np
import os
import cv2
import json
import re

#from os.path import join
imagens = []
tipo = []
caminho = 'UDA-1'
for (dirname, dirs, files) in os.walk(caminho):
    for filename in files:
        if filename.endswith('.jpg'):
           imagens.append(cv2.imread(dirname+"//"+filename,cv2.IMREAD_ANYCOLOR))
        if filename.endswith('.json'):
            a = open(dirname+"//"+filename).read()
            descricao = json.loads(a)
            tipo.append(descricao['meta']['clinical']['benign_malignant'])
            

for t in tipo:
        print(t)
cv2.imshow("imagen",imagens[0])
cv2.waitKey(0)
cv2.destroyAllWindows()