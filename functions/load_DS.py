## LOAD DATASET

## Load the dataset to use in the neural network models. Posible datasets:
## mnist             available in all the models
## fashion_mnist     Not available in Cauchy and Gumbal vgg16 models

## For the categories select a number between 0 and 9. Follow the dataset, the numbers rep´resent a especific category:
## MNIST Categories:
## 0   Class 0
## 1   Class 1
## 2   Class 2
## 3   Class 3
## 4   Class 4
## 5   Class 5
## 6   Class 6
## 7   Class 7
## 8   Class 8
## 9   Class 9
## FASHION-MNIST Categories:
## 0   Class T-shirt/top
## 1   Class Trouser
## 2   Class Pullover
## 3   Class Dress
## 4   Class Coat
## 5   Class Sandal
## 6   Class Shirt
## 7   Class Sneaker
## 8   Class Bag
## 9   Class Ankle boot

def load_ds (DS, Category):

    ## Load Dataset
    import tensorflow_datasets as tfds
    (train_ds,test_ds_train,test_ds_test,test_ds_testadv), INFO = tfds.load(name = DS,
                                    split = ['train[:90%]','test[:90%]','test[90%:99%]','test[99%:]'],
                                    as_supervised = True,
                                    with_info = True)
    len_train=len(train_ds)    ## Info for the nll function

    ## Preprocessing Pipeline
    num_classes =INFO.features["label"].num_classes
    size_img=32
    def img_gen(image, label):
        image = tf.image.resize(image,(size_img,size_img))
        image = tf.cast(image, tf.float32)/255.
        label = tf.one_hot(label, num_classes)
        return image, label
    BATCH_SIZE=64
    clas=Category

    def prepro_db(train_data):
        train_data_ds_tf = train_data.filter(lambda image, label: label == clas).map(img_gen).batch(BATCH_SIZE, drop_remainder = True)
        return(train_data_ds_tf)
    def prepro_dbadv(train_data):
        train_data_ds_tf = train_data.map(img_gen).batch(BATCH_SIZE)
        return(train_data_ds_tf)
    
    ## Apply preprocessing
    train_ds_tf = prepro_db(test_ds_train)
    test_ds_tf = prepro_db(test_ds_test)
    test_ds_testadv = prepro_dbadv(test_ds_testadv)

    return len_train, train_ds_tf, test_ds_tf, test_ds_testadv