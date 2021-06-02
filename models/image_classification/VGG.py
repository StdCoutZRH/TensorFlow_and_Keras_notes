"""使用tf.keras 构建VGG模型

# 引用和参考：
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
- [vgg16.py](
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)
- [vgg19.py](
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *

# Build VGG16 with Keras Sequential Model
def VGG16(input_shape=(224,224,3),classes=1000,include_top=True,weights=None):

    # check weights path
    if weights != None and not os.path.exists(weights):
        raise ValueError("the input of weights is not valid")

    # check include_top and classes
    if include_top and classes!=1000:
        raise ValueError("if include_top=True,classes should be 1000.")

    model = Sequential()

    # Block 1
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=3,strides=1,padding='same',activation='relu',name='block1_conv1'))
    model.add(Conv2D(64,3,strides=1,padding='same',activation='relu',name='block1_conv2'))
    model.add(MaxPooling2D(2,2,'same',name='block1_maxpool'))

    # Block 2
    model.add(Conv2D(128,3,strides=1,padding='same',activation='relu',name='block2_conv1'))
    model.add(Conv2D(128,3,strides=1,padding='same',activation='relu',name='block2_conv2'))
    model.add(MaxPooling2D(2,2,'same',name='block2_maxpool'))

    # Block 3
    model.add(Conv2D(256,3,strides=1,padding='same',activation='relu',name='block3_conv1'))
    model.add(Conv2D(256,3,strides=1,padding='same',activation='relu',name='block3_conv2'))
    model.add(Conv2D(256,3,strides=1,padding='same',activation='relu',name='block3_conv3'))
    model.add(MaxPooling2D(2,2,'same',name='block3_maxpool'))

    # Block 4
    model.add(Conv2D(512,3,strides=1,padding='same',activation='relu',name='block4_conv1'))
    model.add(Conv2D(512,3,strides=1,padding='same',activation='relu',name='block4_conv2'))
    model.add(Conv2D(512,3,strides=1,padding='same',activation='relu',name='block4_conv3'))
    model.add(MaxPooling2D(2,2,'same',name='block4_maxpool'))

    # Block 5
    model.add(Conv2D(512,3,strides=1,padding='same',activation='relu',name='block5_conv1'))
    model.add(Conv2D(512,3,strides=1,padding='same',activation='relu',name='block5_conv2'))
    model.add(Conv2D(512,3,strides=1,padding='same',activation='relu',name='block5_conv3'))
    model.add(MaxPooling2D(2,2,'same',name='block5_maxpool'))

    # include fc layer
    if include_top:
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096,activation='relu',name='fc_layer1'))
        model.add(Dense(4096,activation='relu',name='fc_layer2'))
        model.add(Dense(classes,activation='softmax',name='predictions_layer'))

    # load weights if necessary
    if weights == 'imagenet':
        model.load_weights(weights_path)
        print("Loading weigths from "+weights_path+" finished!")

    return model

# Build VGG19 with Keras Functional API
def VGG19(input_shape=(224,224,3),classes=1000,include_top=True,weights=None):

    # check weights path
    if weights != None and not os.path.exists(weights):
        raise ValueError("the input of weights is not valid")

    # check include_top and classes
    if include_top and classes!=1000:
        raise ValueError("if include_top=True,classes should be 1000.")
    
    input_ = tf.keras.Input(shape=input_shape)
    
    # Block 1
    net = Conv2D(64,3,strides=1,padding='same',activation='relu',name='block1_conv1')(input_)
    net = Conv2D(64,3,strides=1,padding='same',activation='relu',name='block1_conv2')(net)
    net = MaxPooling2D(2,2,'same',name='block1_maxpool')(net)

    # Block 2
    net = Conv2D(128,3,strides=1,padding='same',activation='relu',name='block2_conv1')(net)
    net = Conv2D(128,3,strides=1,padding='same',activation='relu',name='block2_conv2')(net)
    net = MaxPooling2D(2,2,'same',name='block2_maxpool')(net)

    # Block 3
    net = Conv2D(256,3,strides=1,padding='same',activation='relu',name='block3_conv1')(net)
    net = Conv2D(256,3,strides=1,padding='same',activation='relu',name='block3_conv2')(net)
    net = Conv2D(256,3,strides=1,padding='same',activation='relu',name='block3_conv3')(net)
    net = Conv2D(256,3,strides=1,padding='same',activation='relu',name='block3_conv4')(net)
    net = MaxPooling2D(2,2,'same',name='block3_maxpool')(net)

    # Block 4
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block4_conv1')(net)
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block4_conv2')(net)
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block4_conv3')(net)
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block4_conv4')(net)
    net = MaxPooling2D(2,2,'same',name='block4_maxpool')(net)

    # Block 5
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block5_conv1')(net)
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block5_conv2')(net)
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block5_conv3')(net)
    net = Conv2D(512,3,strides=1,padding='same',activation='relu',name='block5_conv4')(net)
    net = MaxPooling2D(2,2,'same',name='block5_maxpool')(net)

    if include_top:
        net = Flatten(name='flatten')(net)
        net = Dense(4096, activation='relu', name='fc1')(net)
        net = Dense(4096, activation='relu', name='fc2')(net)
        net = Dense(classes, activation='softmax', name='predictions')(net)
    else:
        if pooling == 'avg':
            net = GlobalAveragePooling2D()(net)
        elif pooling == 'max':
            net = GlobalMaxPooling2D()(net)

    model = tf.keras.Model(input_, net, name='VGG19')

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
    model = VGG16(weights=None,input_shape=(224,224,3),include_top=True,classes=1000)
    model.summary()
    #   Total params: 138,357,544
    #   Trainable params: 138,357,544
    #   Non-trainable params: 0

    #model = VGG19(weights=None,input_shape=(224,224,3),include_top=True,classes=1000)
    #model.summary()
    #   Total params: 143,667,240
    #   Trainable params: 143,667,240
    #   Non-trainable params: 0