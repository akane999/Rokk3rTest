import keras
from keras.models import model_from_json
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np

model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')
y_pred=[]
X = np.load('X_test.npy')
y_true =np.load('y_test.npy')
for i in range(0,len(X)) :
 y=model.predict(np.array([X[i]]))[0]
 if(y[1]>=0.3):
  y_pred.append(1)
 else :
  y_pred.append(0)
 #print(y)
print(accuracy_score(y_true,y_pred, normalize=True))
print(confusion_matrix(y_true,y_pred))
#print(y_true)



