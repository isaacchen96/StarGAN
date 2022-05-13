import tensorflow as tf
from tensorflow_addons.layers import *
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

class Resblock(Layer):
    def __init__(self, chn, strides):
        super(Resblock, self).__init__()
        self.chn = chn
        self.strides = strides
        self.conv1 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=LeakyReLU(0.3))
        self.bn = BatchNormalization()
        self.conv2 = Conv2D(self.chn, 3, padding='same', activation=LeakyReLU(0.3))
        self.conv3 = Conv2D(self.chn, 3, strides=self.strides, padding='same', activation=None)

    def __call__(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        if inputs.shape[-1] == self.chn:
            return tf.math.add(inputs, conv2)
        else:
            conv3 = self.conv3(inputs)
            return tf.math.add(conv2, conv3)

def build_generator(input_shape=(128,128,1), total_class=2):
    input1 = Input(input_shape)
    input2 = Input(total_class)
    cond = Dense(128, activation=LeakyReLU(0.3))(input2)
    cond = Dense(128*128, activation=LeakyReLU(0.3))(cond)
    cond = Reshape(input_shape)(cond)
    inputs = Concatenate()([input1, cond])
    conv1 = Conv2D(64, 7, strides=(1, 1), padding='same', activation=LeakyReLU(0.3))(inputs)
    bn = BatchNormalization()(conv1)
    conv2 = Conv2D(128, 4, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(conv2)
    conv3 = Conv2D(256, 4, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(conv3)
    for i in range(6):
        resblock = Resblock(256, strides=(1, 1))(bn)
        bn = BatchNormalization()(resblock)
    dconv1 = Conv2DTranspose(128, 4, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(dconv1)
    dconv2 = Conv2DTranspose(64, 4, strides=(2, 2), padding='same', activation=LeakyReLU(0.3))(bn)
    bn = BatchNormalization()(dconv2)
    outputs = Conv2D(1, 7, strides=(1, 1), padding='same', activation='tanh')(bn)

    model = Model([input1, input2], outputs)
    model.summary()

    return model

def build_discriminator(input_shape=(128,128,1)):
    inputs = Input(input_shape)
    conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(inputs)  # 128 -> 64
    drop = Dropout(0.3)(conv1)
    conv2 = Conv2D(128, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(drop)  # 64 -> 32
    drop = Dropout(0.3)(conv2)
    conv3 = Conv2D(256, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(drop)  # 32 -> 16
    drop = Dropout(0.3)(conv3)
    conv4 = Conv2D(512, 3, strides=(2, 2), padding='same', activation=LeakyReLU(0.01))(drop)  # 16 -> 8
    drop = Dropout(0.3)(conv4)

    classified = Conv2D(64, 3, padding='same', activation=LeakyReLU(0.01))(drop)
    drop_c = Dropout(0.3)(classified)
    classified = Conv2D(2, 8, activation=Softmax(axis=-1))(drop_c)
    classified = Flatten()(classified)
    validation = Conv2D(64, 3, strides=(1, 1), padding='same')(conv4)
    drop_v = Dropout(0.3)(validation)
    validation = Conv2D(2, 3, strides=(1, 1), padding='same', activation=Softmax(axis=-1))(drop_v)
    model = Model(inputs, [validation, classified])
    model.summary()
    return model

