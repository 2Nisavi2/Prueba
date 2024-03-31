## --LOAD MODELS-- ##
## This code allow the load of the model follow the kind of neural network. This is the list of possible models to load:
# 'vgg16_mnist'             Deterministic vgg16 model in mnist dataset
# 'vgg19_mnist'             Deterministic vgg19 model in mnist dataset
# 'vgg16bt_mnist'           Full MNF Bayesian vgg16 model in mnist dataset
# 'vgg19bt_mnist'           Full MNF Bayesian vgg19 model in mnist dataset
# 'vgg16b1_mnist'           Last layer MNF Bayesian vgg16 model in mnist dataset
# 'vgg19b1_mnist'           Last layer MNF Bayesian vgg19 model in mnist dataset
# 'vgg16bt_Re_mnist'        Full Reparameterization Trick Bayesian vgg16 model in mnist dataset
# 'vgg19bt_Re_mnist'        Full Reparameterization Trick Bayesian vgg19 model in mnist dataset
# 'vgg16b1_Re_mnist'        Last layer Reparameterization Trick Bayesian vgg16 model in mnist dataset
# 'vgg19b1_Re_mnist'        Last layer Reparameterization Trick Bayesian vgg19 model in mnist dataset
# 'vgg16_fashion'           Deterministic vgg16 model in fashion_mnist dataset
# 'vgg19_fashion'           Deterministic vgg19 model in fashion_mnist dataset
# 'vgg16bt_fashion'         Full MNF Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19bt_fashion'         Full MNF Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16b1_fashion'         Last layer MNF Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19b1_fashion'         Last layer MNF Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16bt_Re_fashion'      Full Reparameterization Trick Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19bt_Re_fashion'      Full Reparameterization Trick Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16b1_Re_fashion'      Last layer Reparameterization Trick Bayesian vgg16 model in fashion_mnist dataset
# 'vgg19b1_Re_fashion'      Last layer Reparameterization Trick Bayesian vgg19 model in fashion_mnist dataset
# 'vgg16bt_CA'              Full Cauchy MNF Bayesian vgg16 model in mnist dataset
# 'vgg16b1_CA'              Full Cauchy MNF Bayesian vgg16 model in mnist dataset
# 'vgg16bt_GUM'             Full Gumbel MNF Bayesian vgg16 model in mnist dataset
# 'vgg16b1_GUM'             Full Cumbel MNF Bayesian vgg16 model in mnist dataset

def load_model (KIND, ARC, DATA):

    if KIND == 'DET':
        from Prueba.networks import vgg as mod
        if ARC == 'vgg16':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16/fashion_mnist/checkpoint'
        if ARC == 'vgg19':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19/fashion_mnist/checkpoint'
    if KIND == 'MNF_1C':
        from Prueba.networks import vgg_b1 as mod
        if ARC == 'vgg16':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16b1/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16b1/fashion_mnist/checkpoint'
        if ARC == 'vgg19':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19b1/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19b1/fashion_mnist/checkpoint'
    if KIND == 'MNF_BT':
        from Prueba.networks import vgg_bt as mod
        if ARC == 'vgg16':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16bt/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16bt/fashion_mnist/checkpoint'
        if ARC == 'vgg19':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19bt/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19bt/fashion_mnist/checkpoint'
    if KIND == 'REP_1C':
        from Prueba.networks import vgg_b1_Re as mod
        if ARC == 'vgg16':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16b1Re/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16b1Re/fashion_mnist/checkpoint'
        if ARC == 'vgg19':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19b1Re/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19b1Re/fashion_mnist/checkpoint'
    if KIND == 'REP_BT':
        from Prueba.networks import vgg_bt_Re as mod
        if ARC == 'vgg16':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16btRe/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg16'
                WEIGHT = 'Prueba/weights/vgg16btRe/fashion_mnist/checkpoint'
        if ARC == 'vgg19':
            if DATA == 'mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19btRe/mnist/checkpoint'
            if DATA == 'fashion_mnist':
                BUILD_CAT = 'vgg19'
                WEIGHT = 'Prueba/weights/vgg19btRe/fashion_mnist/checkpoint'
    if KIND == 'CAU_1C':
        from Prueba.networks import vgg_b1_MNF_CA as mod
        if ARC == 'vgg16':
            BUILD_CAT = 'vgg16'
            WEIGHT = 'Prueba/weights/vgg16MNFLUb1_CA/mnist/checkpoint'
    if KIND == 'CAU_BT':
        from Prueba.networks import vgg_bt_MNF_CA as mod
        if ARC == 'vgg16':
            BUILD_CAT = 'vgg16'
            WEIGHT = 'Prueba/weights/vgg16MNFLUbt_CA/mnist/checkpoint'
    if KIND == 'GUM_1C':
        from Prueba.networks import vgg_b1_MNF_GUM as mod
        if ARC == 'vgg16':
            BUILD_CAT = 'vgg16'
            WEIGHT = 'Prueba/weights/vgg16MNFLUb1_GUM/mnist/checkpoint'
    if KIND == 'GUM_BT':
        from Prueba.networks import vgg_bt_MNF_GUM as mod
        if ARC == 'vgg16':
            BUILD_CAT = 'vgg16'
            WEIGHT = 'Prueba/weights/vgg16MNFLUbt_GUM/mnist/checkpoint'

    BUILD_NUM = 10
    TENSOR = (None,32,32,1)
        
    return BUILD_CAT, BUILD_NUM, TENSOR, WEIGHT