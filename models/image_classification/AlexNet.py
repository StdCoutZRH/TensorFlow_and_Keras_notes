"""使用tf.keras functional api构建AlexNet模型

# 引用和参考：
- [ImageNet classification with deep convolutional neural networks](
    https://www.onacademic.com/detail/journal_1000039913864210_2a08.html) 

"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *

# Build AlexNet with Keras Functional API
def AlexNet(input_shape=(224,224,3),classes=1000,include_top=True,weights=None):

    # check weights path
    if weights != None and not os.path.exists(weights):
        raise ValueError("the input of weights is not valid")

    # check include_top and classes
    if include_top and classes!=1000:
        raise ValueError("if include_top=True,classes should be 1000.")

    input_ = tf.keras.Input(shape=input_shape,dtype=tf.float32)
    net = ZeroPadding2D(padding=((1,2),(1,2)))(input_)
    
    # first conv layer
    net = Conv2D(filters=96,kernel_size=11,strides=4,padding='valid',activation='relu',name='conv_1')(net)
    net = BatchNormalization(axis=1)(net)
    net = MaxPool2D(pool_size=3,strides=2,padding='valid',name='maxpool_1')(net)  

    # second conv layer
    net = Conv2D(filters=256,kernel_size=5,strides=1,padding='same',activation='relu',name='conv_2')(net)
    net = BatchNormalization(axis=1)(net)
    net = MaxPool2D(3,2,padding='valid',name='maxpool_2')(net)

    # third conv layer
    net = Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_3')(net)

    # forth and fifth conv layer
    net = Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_4')(net)
    net = Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_5')(net)
    net = MaxPool2D(3,2,padding='valid',name='maxpool3')(net)
    
    if include_top:
        net = Flatten(name='flatten')(net)
        net = Dense(4096, activation='relu',name='fc1')(net)
        net = Dropout(0.5,name='dropout_1')(net)
        net = Dense(4096, activation='relu',name='fc2')(net)
        net = Dropout(0.5,name='dropout_2')(net)
        net = Dense(classes, activation='softmax',name='predictions')(net)

    model = tf.keras.Model(input_, net, name='AlexNet')

    # load weights if necessary
    if weights != None:
        model.load_weights(weights)
        print("Loading weigths from "+weights+" finished!")

    return model

if __name__=='__main__':

    # set env and gpu
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'

    phy_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in phy_gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

    # test model
    model = AlexNet(weights=None,input_shape=(224,224,3),include_top=True,classes=1000)
    model.summary()

    #   =================================================================
    #   Total params: 62,378,672
    #   Trainable params: 62,378,508
    #   Non-trainable params: 164