"""使用tf.keras sequential model构建VGG16模型，使用functional api构建VGG19模型

# 引用和参考：
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
- [vgg16.py](
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py)
- [vgg19.py](
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *

import os

# 权重文件的下载链接
VGG16_WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
VGG16_WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
VGG19_WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
VGG19_WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

def vgg16(include_top=True,weights='imagenet',
                    input_shape=None,pooling=None,classes=1000):
    """使用tf.keras sequential model构建VGG16模型

    # Arguments
        include_top:是否包含网络最后的3层全连接层，默认为包含。
        weights:选择预训练权重，默认为'imagenet',可选'None'为随机初始化权重或者其他权重的路径。
        input_shape:输入的尺寸，应该是一个元组，当include_top设为True时默认为(224,224,3)，否则应当被
                    定制化，因为输入图像的尺寸会影响到全连接层参数的个数。
        pooling:指定池化方式。
        classes:类别数量。
    # Returns
        返回一个tf.keras sequential model实例。
    # Raises
        ValueError：由于不合法的参数会导致相应的异常。
    """

    # 检测weights参数是否合法
    if not(weights in {'imagenet',None} or os.path.exists(weights)):
        raise ValueError("the input of weights is not valid")

    # 检测include_top和classes是否冲突
    if weights=='imagenet' and include_top and classes!=1000:
        raise ValueError("if using weights='imagenet' and include_top=True,classes should be 1000.")

    model = Sequential()

    # Block 1
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=3,
                            strides=1,padding='same',activation='relu',name='block1_conv1'))
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

    if include_top: #包含默认的全连接层
        model.add(Flatten(name='flatten'))
        model.add(Dense(4096,activation='relu',name='fc_layer1'))
        model.add(Dense(4096,activation='relu',name='fc_layer2'))
        model.add(Dense(classes,activation='softmax',name='predictions_layer'))
    else:
        if pooling == 'avg':
            model.add(GlobalAveragePooling2D())
        elif pooling == 'max':
            model.add(GlobalMaxPooling2D())

    # 加载权重
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                VGG16_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='64373286793e3c8b2b4e3219cbf3544b')
        else:
            weights_path = get_file(
                'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                VGG16_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path)
        print("Loading weigths from "+weights_path+" finished!")
    elif weights is not None:
        model.load_weights(weights)
        print("Loading weigths from "+weights+" finished!")

    return model

def vgg19(include_top=True,weights='imagenet',
                    input_shape=None,pooling=None,classes=1000):
    """使用tf.keras functional api构建VGG19模型

    # Arguments
        include_top:是否包含网络最后的3层全连接层，默认为包含。
        weights:选择预训练权重，默认为'imagenet',可选'None'为随机初始化权重或者其他权重的路径。
        input_shape:输入的尺寸，应该是一个元组，当include_top设为True时默认为(224,224,3)，否则应当被
                    定制化，因为输入图像的尺寸会影响到全连接层参数的个数。
        pooling:指定池化方式。
        classes:类别数量。
    # Returns
        返回一个tf.keras model实例。
    # Raises
        ValueError：由于不合法的参数会导致相应的异常。
    """

    # 检测weights参数是否合法
    if not(weights in {'imagenet',None} or os.path.exists(weights)):
        raise ValueError("the input of weights is not valid")

    # 检测include_top和classes是否冲突
    if weights=='imagenet' and include_top and classes!=1000:
        raise ValueError("if using weights='imagenet' and include_top=True,classes should be 1000.")
    
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

    # 加载权重
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                VGG19_WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                VGG19_WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        print("Loading weigths from "+weights_path+" finished!")
    elif weights is not None:
        model.load_weights(weights)
        print("Loading weigths from "+weights+" finished!")

    return model

if __name__=='__main__':
    
    # set gpu and env
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf 
    phy_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in phy_gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    
    # test
    #model = vgg16(weights='imagenet',input_shape=(224,224,3),include_top=True,classes=100)
    #model.summary()
    #   Total params: 138,357,544
    #   Trainable params: 138,357,544
    #   Non-trainable params: 0

    model = vgg19(weights=None,input_shape=(224,224,3),include_top=True,classes=1000)
    model.summary()
    #   Total params: 143,667,240
    #   Trainable params: 143,667,240
    #   Non-trainable params: 0