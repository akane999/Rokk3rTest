import keras
from keras.models import model_from_json
model = model_from_json(open('model.json').read())
model.load_weights('weights.h5')
import numpy as np
def print_indicator(data, model, class_names, bar_width=50):
    probabilities = model.predict(np.array([data]))[0]
    left_side = '-' * int(probabilities[1] * bar_width)
    right_side = '-' * int(probabilities[0] * bar_width)
    print (class_names[0], left_side + '###' + right_side, class_names[1])

X = np.load('X_test.npy')
class_names = ['noteeth', 'teeth']
for i in range(0,len(X)) :
 print_indicator(X[i], model, class_names)





