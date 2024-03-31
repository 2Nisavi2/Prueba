## LOAD DATASET

## Load the dataset to use in the neural network models. Posible datasets:
## mnist             available in all the models
## fashion_mnist     Not available in Cauchy and Gumbal vgg16 models

def load_ds (DS):
    (train_ds,test_ds_train,test_ds_test,test_ds_testadv), INFO = tfds.load(name = DS,
                                split = ['train[:90%]','test[:90%]','test[90%:99%]','test[99%:]'],
                                as_supervised = True,
                                with_info = True)
    len_train=len(train_ds)

    num_classes =INFO.features["label"].num_classes
    size_img=32
    return train_ds, test_ds_train, test_ds_test, test_ds_testadv, len_train, num_classes, size_img