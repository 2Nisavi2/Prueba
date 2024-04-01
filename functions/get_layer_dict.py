import tensorflow as tf

def get_layer_dictionary(model,input_shape=(32,32,1)):
    dict_models={}
    input=xvar=tf.keras.Input(input_shape)
    for idx,layer in enumerate(model.layers):
        xvar=layer(xvar)
        dict_models[layer.name]=tf.keras.Model(input,xvar,name=layer.name)
        print(idx,layer.name)
    return dict_models