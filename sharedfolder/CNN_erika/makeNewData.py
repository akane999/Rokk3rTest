import glob
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
positive_paths = glob.glob(os.path.join("positivos64", "*.png"))
print ('loaded', len(positive_paths), 'positive_examples')
examples =  [(path, 1) for path in positive_paths]

import numpy as np
from skimage.measure import block_reduce
from skimage.io import imread
block_size=2
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
for path, label in examples:
        img = imread(path, as_grey=True)
        #img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
        x = img_to_array(img)  
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1,
         save_to_dir='positivos64', save_prefix='positivas', save_format='png'):
         i += 1
         if i > 12:
          break
