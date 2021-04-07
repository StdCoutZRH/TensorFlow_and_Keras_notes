"""使用tf.keras functional api构建AlexNet模型

# 引用和参考：
- [ImageNet classification with deep convolutional neural networks](
    https://www.onacademic.com/detail/journal_1000039913864210_2a08.html) 

"""

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
import os

def AlexNet(input_shape=(224,224,3),classes=1000,include_top=True,weights=None):

    """使用tf.keras functional api构建AlexNet模型

    # Arguments  
        input_shape:输入的尺寸，应该是一个元组，当include_top设为True时默认为(224,224,3)，否则应当被
                    定制化，因为输入图像的尺寸会影响到全连接层参数的个数。
        classes:类别数量。
        include_top:是否包含网络最后的3层全连接层，默认为包含。
        weights:选择预训练权重，默认'None'为随机初始化权重。
    # Returns
        返回一个tf.keras Model实例。
    # Raises
        ValueError：由于不合法的参数会导致相应的异常。
    """

    # 检测weights参数是否合法
    if weights != None and not os.path.exists(weights):
        raise ValueError("the input of weights is not valid")

    input_img = tf.keras.Input(shape=input_shape,dtype=tf.float32)  #input shape = (None,224,224,3)
    net = ZeroPadding2D(padding=((1,2),(1,2)))(input_img)   #(None,224,224,3)->(None,227,227,3)
    
    # first conv layer
    net = Conv2D(filters=96,kernel_size=11,strides=4,padding='valid',activation='relu',name='conv_1')(net)   #(None,227,227,3)->(None,55,55,96)
    net = BatchNormalization(axis=1)(net)
    net = MaxPooling2D(pool_size=3,strides=2,padding='same',name='maxpool_1')(net)  #(None,55,55,96)->(None,28,28,96)

    # second conv layer
    net = Conv2D(filters=256,kernel_size=5,strides=1,padding='same',activation='relu',name='conv_2')(net)   #(None,28,28,96)->(None,,28,28,256)
    net = BatchNormalization(axis=1)(net)
    net = MaxPooling2D(3,2,padding='valid',name='maxpool_2')(net)   #(None,28,28,256)->(None,13,13,256)

    # third conv layer
    net = Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_3')(net)   #(None,13,13,256)->(None,13,13,384)

    # forth and fifth conv layer
    net = Conv2D(filters=384,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_4')(net)   #(None,13,13,384)->(None,13,13,384)
    net = Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_5')(net)   #(None,13,13,384)->(None,13,13,256)

    net = MaxPooling2D(3,2,padding='valid',name='maxpool3')(net)    #(None,13,13,256)->(None,6,6,256)
    
    if include_top:
        net = Flatten(name='flatten')(net)  #(None,6,6,256)->(None, 9216) 
        net = Dense(4096, activation='relu', name='fc1')(net)   #(None, 9216)->(None,4096)
        net = Dropout(0.5,name='dropout_1')(net)
        net = Dense(4096, activation='relu', name='fc2')(net)   #(None, 4096)->(None,4096)
        net = Dropout(0.5,name='dropout_2')(net)
        net = Dense(classes, activation='softmax', name='predictions')(net) #(None, 4096)->(None,1000)

    model = tf.keras.Model(input_img, net, name='AlexNet')

    # 加载权重
    if weights != None:
        model.load_weights(weights)
        print("Loading weigths from "+weights+" finished!")

    return model

if __name__=='__main__':
    
    # set gpu and env
    os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
    #os.environ['CUDA_VISIBLE_DEVICES']='0'
    import tensorflow as tf 
    phy_gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in phy_gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    
    # test
    model = AlexNet(weights=None,input_shape=(224,224,3),include_top=True,classes=1000)
    model.summary()
    #   =================================================================
    #   Total params: 62,378,676
    #   Trainable params: 62,378,510
    #   Non-trainable params: 166