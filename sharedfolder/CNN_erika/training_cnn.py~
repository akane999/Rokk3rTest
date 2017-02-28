import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
# load the data
X = np.load('X_train.npy')
y = np.load('y_train.npy')


nb_classes = 2
y = np_utils.to_categorical(y, nb_classes).astype(np.float32)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# balance classes because are unbalanced
class_totals = y.sum(axis=0)
class_weight = class_totals.max() / class_totals

print (X.dtype, X.min(), X.max(), X.shape)
print (y.dtype, y.min(), y.max(), y.shape)
img_rows, img_cols = X.shape[1:]
nb_filters = 32
nb_pool = 2
nb_conv = 3



####Convolutional network architecture
model = Sequential()
model.add(Reshape((1,img_rows, img_cols), input_shape=(img_rows, img_cols)))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Convolution2D(16, nb_conv, nb_conv, activation='relu'))
model.add(Convolution2D(16, nb_conv, nb_conv, activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
#####

adam=Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
validation_split = 0.10
model.fit(X, y, batch_size=128, class_weight=class_weight, nb_epoch=70, verbose=1, validation_split=validation_split)
open('model.json', 'w').write(model.to_json())
model.save_weights('weights.h5')


plt.plot(model.model.history.history['loss'])
#plt.plot(model.model.history.history['acc'])
#plt.plot(model.model.history.history['val_loss'])
plt.plot(model.model.history.history['val_acc'])
plt.show()

n_validation = int(len(X) * validation_split)
y_predicted = model.predict(X[-n_validation:])
print (roc_auc_score(y[-n_validation:], y_predicted))
