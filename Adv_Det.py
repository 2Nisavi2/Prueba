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
from Prueba.functions.load_weight import load_weight

### --LOAD DATASET-- ###
print('Loading dataset')
data = load_ds(DATA, VICTIM_CLASS)
print('Done load')

### --LOAD ARCHITECTURE-- ###
if KIND == 'DET':
    from Prueba.networks.vgg import VGG as mod
if KIND == 'MNF_1C':
    from Prueba.networks.vgg_b1 import VGG as mod
if KIND == 'MNF_BT':
    from Prueba.networks.vgg_bt import VGG as mod
if KIND == 'REP_1C':
    from Prueba.networks.vgg_b1_Re import VGG as mod
if KIND == 'REP_BT':
    from Prueba.networks.vgg_bt_Re import VGG as mod
if KIND == 'CAU_1C':
    from Prueba.networks.vgg_b1_MNF_CA import VGG as mod
if KIND == 'CAU_BT':
    from Prueba.networks.vgg_bt_MNF_CA import VGG as mod
if KIND == 'GUM_1C':
    from Prueba.networks.vgg_b1_MNF_GUM import VGG as mod
if KIND == 'GUM_BT':
    from Prueba.networks.vgg_bt_MNF_GUM import VGG as mod

### --LOAD WEIGHTS-- ###
print('Loading weights')
Model = load_weight(KIND, ARC, DATA)
print('Weights loaded')

### --COMPILE MODEL-- ###
print('Compile model')
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

print('Model compiled')

### --TEST MODEL-- ###
print('Testing model')
model.evaluate(data[1])