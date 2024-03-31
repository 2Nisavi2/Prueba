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

def load_model (mod):
    if mod == 'vgg16_mist':
        from Prueba.networks import VGG
        model = VGG('vgg16', 10)
        model.build((None,32,32,1))
        model.load_weights('Prueba/weights/vgg16/mnist/checkpoint')
        def nll(y_true, y_pred):
            cross_entropy=-y_pred.log_prob(y_true)
            nll = tf.reduce_mean(cross_entropy)+model.kl_div() / data[0]
            return nll
        model.compile(optimizer="adam",
                loss=nll,
                metrics=["accuracy"])
        