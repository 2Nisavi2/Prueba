## --Load Libraries-- ##
import argparse
import pathlib
from tensorflow.keras import layers
import tensorflow as tf
import sys
import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy.stats import wasserstein_distance,energy_distance
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt

### --PARAMETERS-- ###
ARC ='vgg16'
KIND = 'DET'
DATA = 'mnist'
VICTIM_CLASS = 5
ADV_CLASS = 8

### --Load Functions-- ##
from Prueba.functions.load_DS import load_ds
from Prueba.functions.load_model import load_model

### --LOAD DATASET-- ###
## Load the dataset to use in the neural network models. Posible datasets:
# 'mnist'             Available in all the models
# 'fashion_mnist'     Not available in Cauchy and Gumbal vgg16 models

## For the categories select a number between 0 and 9. Follow the dataset, the numbers rep´resent a especific category:
## MNIST Categories:
# 0   Class 0
# 1   Class 1
# 2   Class 2
# 3   Class 3
# 4   Class 4
# 5   Class 5
# 6   Class 6
# 7   Class 7
# 8   Class 8
# 9   Class 9
## FASHION-MNIST Categories:
# 0   Class T-shirt/top
# 1   Class Trouser
# 2   Class Pullover
# 3   Class Dress
# 4   Class Coat
# 5   Class Sandal
# 6   Class Shirt
# 7   Class Sneaker
# 8   Class Bag
# 9   Class Ankle boot
print('Loading dataset')
data = load_ds(DATA, VICTIM_CLASS)
print('Done load')

## --LOAD MODELS-- ##
print('Loading model')
Model = load_model(KIND, ARC, DATA)    
model = mod(Model[0], Model[1])
model.build(Model[2])
model.load_weights(Model[3])

## Loss Function
if KIND =='DET':
    loss = tf.keras.losses.CategoricalCrossentropy()
else:
    def nll(y_true, y_pred):
        cross_entropy=-y_pred.log_prob(y_true)
        nll = tf.reduce_mean(cross_entropy)+model.kl_div() / data[0]
        return nll
    loss = nll

model.compile(optimizer="adam",
        loss=loss,
        metrics=["accuracy"])

print('Model load')

### --TEST MODEL-- ###
print('Testing model')
model.evaluate(data[1])















MOD = 'vgg16_mnist'
Model = load_model(MOD)

