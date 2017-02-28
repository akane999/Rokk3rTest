#This program requires python3. and keras with theano backend.
# Execute this program to classify images.

import cv2
import numpy as np
import os
from skimage import io
import keras
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
from skimage.transform import resize
from skimage.measure import block_reduce
from shutil import copy2


#load model
model = model_from_json(open('model_final.json').read())
model.load_weights('weights_final.h5')

DatasetPath = []
block_size=2
#load data path
for i in os.listdir("muct-b-jpg-v1"):
    DatasetPath.append(os.path.join("muct-b-jpg-v1", i))
imageData = []
imageLabels = []
X = []
# for each file in the path classify and copy
for i in DatasetPath:
    imgRead = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))# equalization to improve quality
    imgReadeq = clahe.apply(imgRead)
    labelRead = os.path.split(i)[1].split(".")[0]
    faceDetectClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#face detection haar
    faces = faceDetectClassifier.detectMultiScale(imgReadeq,1.2,3,0)
    if len(faces)>0:
     for (x, y, w, h) in faces:
         if w>0:
          roi_gray = cv2.resize(imgReadeq[y:y + h, x:x + w],(64,64)) 
          img = block_reduce(roi_gray, block_size=(block_size, block_size), func=np.mean)#resize 32x32 
          #cv2.imwrite("test/"+labelRead+".png",img);
          X = np.asarray(img)
          X = X.astype(np.float32) / 255.
          y = model.predict(np.array([X]))[0] #model prediction
          if (y[1]>=0.3):
           copy2(i, "/opt/datos/rokk3r/pos/")
          else:
           copy2(i, "/opt/datos/rokk3r/neg/")
           
          print(labelRead+" , "+str(y))
        

