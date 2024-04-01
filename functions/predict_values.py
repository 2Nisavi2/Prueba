### --GET MODELS LAYERS DICTIONARY-- ###

import numpy as np
import tensorflow as tf
from Prueba.functions.get_layer_dict import get_layer_dictionary as gld

def get_pred (model, KIND, BLOCK, BASE_DATA, TEST_DATA, ADV_DATA):
    
    dict_model = gld(model, input_shape=(32,32,1))

    ## Layers Name by Architecture and Block
    if KIND == 'DET':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'dense'
    if KIND == 'MNF_1C':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'mnf_dense'
    if KIND == 'MNF_BT':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'mnf_dense'
    if KIND == 'REP_1C':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'dense_reparameterization'
    if KIND == 'REP_BT':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'dense'
    if KIND == 'CAU_1C':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'mnf_dense'
    if KIND == 'CAU_BT':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'mnf_dense'
    if KIND == 'GUM_1C':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'mnf_dense'
    if KIND == 'GUM_BT':
        if BLOCK == 'SEQ':
            CAP = 'sequential'
        if BLOCK == 'DEN':
            CAP = 'mnf_dense'

    ## Save Numpy data about base predict, test predict and adversarial predict
    # Predict Base images
    print('Predict base images')
    valueb_list=[]
    for databx, databy in BASE_DATA:
        for _ in range(10):
            predictbx=dict_model[CAP](databx)
            valueb_list.append(predictbx)
    valueb_list= np.concatenate(valueb_list)
    folder_name = 'Prueba/generate_files/base/{}'.format('Base')
    np.save(folder_name, valueb_list)
    print('Base predict saved')

    # Predict Test images
    print('Predict test images')
    valueb_list_test=[]
    for dataxtest, dataytest in TEST_DATA.take(1):
        dataxtest1=dataxtest[:1] ### Just take the first image
        for _ in range(20):
            predictx=dict_model[CAP](dataxtest1)
            valueb_list_test.append(predictx)
    valueb_list_test= np.concatenate(valueb_list_test)
    folder_name = 'Prueba/generate_files/test/{}'.format('Test')
    np.save(folder_name, valueb_list_test)
    print('Test predict saved')