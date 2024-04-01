### --ADVERSARIAL ATTACK GENERATOR-- ###
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Prueba.functions.get_layer_dict import get_layer_dictionary as gld

def adv_att_gen (model, ADV_DATA, ADV_CLASS, VICTIM_CLASS, DATA, ATTACK, EPX, KIND, BLOCK):
    if ATTACK == 'PGD':
        from Prueba.functions.load_attack import pgd_attack as att
    if ATTACK == 'CW':
        from Prueba.functions.load_attack import cw_attack as att

    ## Victim Image Ubication in Adversarial dataset
    if DATA == 'mnist':
        if ADV_CLASS == 0:
            Vic_loc = 5
        if ADV_CLASS == 1:
            Vic_loc = 1
        if ADV_CLASS == 2:
            Vic_loc = 3
        if ADV_CLASS == 3:
            Vic_loc = 0
        if ADV_CLASS == 4:
            Vic_loc = 2
        if ADV_CLASS == 5:
            Vic_loc = 7
        if ADV_CLASS == 6:
            Vic_loc = 11
        if ADV_CLASS == 7:
            Vic_loc = 21
        if ADV_CLASS == 8:
            Vic_loc = 8
        if ADV_CLASS == 9:
            Vic_loc = 20
    if DATA == 'fashion_mnist':
        if ADV_CLASS == 0:
            Vic_loc = 10
        if ADV_CLASS == 1:
            Vic_loc = 8
        if ADV_CLASS == 2:
            Vic_loc = 25
        if ADV_CLASS == 3:
            Vic_loc = 36
        if ADV_CLASS == 4:
            Vic_loc = 0
        if ADV_CLASS == 5:
            Vic_loc = 7
        if ADV_CLASS == 6:
            Vic_loc = 1
        if ADV_CLASS == 7:
            Vic_loc = 4
        if ADV_CLASS == 8:
            Vic_loc = 28
        if ADV_CLASS == 9:
            Vic_loc = 3

    ## Adversarial Tensor
    if DATA == 'mnist':
        if VICTIM_CLASS == 0:
            target_label = np.array([1,0,0,0,0,0,0,0,0,0])
        if VICTIM_CLASS == 1:
            target_label = np.array([0,1,0,0,0,0,0,0,0,0])
        if VICTIM_CLASS == 2:
            target_label = np.array([0,0,1,0,0,0,0,0,0,0])
        if VICTIM_CLASS == 3:
            target_label = np.array([0,0,0,1,0,0,0,0,0,0])
        if VICTIM_CLASS == 4:
            target_label = np.array([0,0,0,0,1,0,0,0,0,0])
        if VICTIM_CLASS == 5:
            target_label = np.array([0,0,0,0,0,1,0,0,0,0])
        if VICTIM_CLASS == 6:
            target_label = np.array([0,0,0,0,0,0,1,0,0,0])
        if VICTIM_CLASS == 7:
            target_label = np.array([0,0,0,0,0,0,0,1,0,0])
        if VICTIM_CLASS == 8:
            target_label = np.array([0,0,0,0,0,0,0,0,1,0])
        if VICTIM_CLASS == 9:
            target_label = np.array([0,0,0,0,0,0,0,0,0,1])

    if DATA == 'fashion_mnist':
        if VICTIM_CLASS == 0:
            target_label = np.array([1,0,0,0,0,0,0,0,0,0])
        if VICTIM_CLASS == 1:
            target_label = np.array([0,1,0,0,0,0,0,0,0,0])
        if VICTIM_CLASS == 2:
            target_label = np.array([0,0,1,0,0,0,0,0,0,0])
        if VICTIM_CLASS == 3:
            target_label = np.array([0,0,0,1,0,0,0,0,0,0])
        if VICTIM_CLASS == 4:
            target_label = np.array([0,0,0,0,1,0,0,0,0,0])
        if VICTIM_CLASS == 5:
            target_label = np.array([0,0,0,0,0,1,0,0,0,0])
        if VICTIM_CLASS == 6:
            target_label = np.array([0,0,0,0,0,0,1,0,0,0])
        if VICTIM_CLASS == 7:
            target_label = np.array([0,0,0,0,0,0,0,1,0,0])
        if VICTIM_CLASS == 8:
            target_label = np.array([0,0,0,0,0,0,0,0,1,0])
        if VICTIM_CLASS == 9:
            target_label = np.array([0,0,0,0,0,0,0,0,0,1])

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

    for dsadv_testx,dsadv_testy in ADV_DATA.take(1):
        dsadv_testx1=dsadv_testx[Vic_loc]
        print('Natural Image')
        plt.imshow(dsadv_testx1)
        labeltrue=dsadv_testy[Vic_loc]
        img_adv=att(dict_model[CAP], dsadv_testx1, target_label, epsilon=EPX, num_steps=200, step_size=0.01)
        print('Adversarial Image')
        plt.imshow(img_adv)

    dict_model = gld(model, input_shape=(32,32,1))

    print('Predict Adv images')
    valueb_list_testadv=[]
    for _ in range(20):
            predictx=dict_model[CAP](img_adv[None,...])
            valueb_list_testadv.append(predictx)
    valueb_list_testadv= np.concatenate(valueb_list_testadv)
    folder_name = 'Prueba/generate_files/adv/{}'.format('Adv')
    np.save(folder_name, valueb_list_testadv)
    print('Adv predict saved')