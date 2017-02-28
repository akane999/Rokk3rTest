# Rokk3rTest
Rokk3r labs test

by Erika Torres.

Steps to solve teeth detection with Neural Networks

1. Equalize histogram of images in grayscale using 8x8 blocks because illumination is not homogeneus.
2. Use cascade filter with haar transform to detect faces, this function is already available in openCV.
3. Crop image with the results of step 2 , resize and force to 64x64.
4. Block reduce image in order to be used in a DNN.
5. Classify manually the images present in the first folder between teeth and no teeth.The folders are in CNN_erika      positivos64 and negativos64.
6. Create more data artificially because is insufficient to train a DNN (CNN_erika/training.py)
7. Design a CNN and improve it changing the hyperparameters. The model is represented by weight.h5 and model.json
8. Classify images with the DNN of step 7. Run classifier.py

To run the test go to folder rokk3r and run 

python3.4 classifier.py

The results can be checked in the file resultados.

CONTENTS IN FOLDERS

CNN_erika: All files and programs used to train CNN for teeth detection. Use a GPU if you want to try this code.
rokk3r: All files and programs used to classify the photos.

REQUIREMENTS
1. Python3
2. Scikit-image
3. opencv-python
4. h5py
5. keras
6. theano

**Convolutional Network classifier

The expected error is 11% with this CNN. It was trained with 4229 positive examples and 3166 negative examples , I used makeNewData.py to generate artificial data
to reduce the overfitting. I did a validation run with 10% of the samples.

The architecture used was 
 2 layer conv
 maxpool
 2 layer conv
 maxpool
 dropout(0.25)
 flatten
 dense(128,relu)
 dropout(0.5)
 dense(2,sofmax)

This code can be improved using better techniques to detect the face, because haar presents a lot of false positives. The next option is using HOG features.
