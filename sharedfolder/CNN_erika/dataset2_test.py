import glob
import os
positive_paths = glob.glob(os.path.join("smiles", "*.png"))
print ('loaded', len(positive_paths), 'positive examples')
examples = [(path, 0) for path in positive_paths]
#+ [(path, 0) for path in negative_paths] 

import numpy as np
from skimage.measure import block_reduce
from skimage.io import imread

def examples_to_dataset(examples, block_size=2):
    X = []
    y = []
    for path, label in examples:
        img = imread(path, as_grey=True)
        img = block_reduce(img, block_size=(block_size, block_size), func=np.mean)
        X.append(img)
        y.append(label)
    return np.asarray(X), np.asarray(y)

X, y = examples_to_dataset(examples)

X = X.astype(np.float32) / 255.
y = y.astype(np.int32)
print (X.dtype, X.min(), X.max(), X.shape)
print (y.dtype, y.min(), y.max(), y.shape)
#from utils import make_mosaic, show_array
#show_array(255 * make_mosaic(X[:len(negative_paths)], 8), fmt='jpeg') # negative at the beginning
#show_array(255 * make_mosaic(X[-len(positive_paths):], 8), fmt='jpeg') # positive at the end
np.save('X_test.npy', X)
np.save('y_test.npy', y)
