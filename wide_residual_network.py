# This code is imported and adapted from the following project: https://github.com/titu1994/Wide-Residual-Networks

from keras.models import Model
from keras.layers import Input, concatenate, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization

from keras import backend as K

from keras.regularizers import Regularizer
import numpy as np

def initial_conv(input):
    x = Conv2D(16, (3, 3), padding="same")(input)

    channel_axis = 1 if K.image_data_format() == "th" else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def conv1_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_data_format() == "th" else -1

    # Check if input number of filters is same as 16 * k, else create Conv2D for this input
    if K.image_data_format() == "th":
        if init._keras_shape[1] != 16 * k:
            init = Conv2D(160, (1, 1), activation="linear", padding="same")(init)
    else:
        if init._keras_shape[-1] != 16 * k:
            init = Conv2D(160, (1, 1), activation="linear", padding="same")(init)

    x = Conv2D(160, (3, 3), padding="same")(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(160, (3, 3), padding="same", kernel_regularizer=None)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = concatenate([init, x])
    return m

def conv2_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_data_format() == "th" else -1

    # Check if input number of filters is same as 32 * k, else create Conv2D for this input
    if K.image_data_format() == "th":
        if init._keras_shape[1] != 32 * k:
            init = Conv2D(320, (1, 1), activation="linear", padding="same")(init)
    else:
        if init._keras_shape[-1] != 32 * k:
            init = Conv2D(320, (1, 1), activation="linear", padding="same")(init)

    x = Conv2D(320, (3, 3), padding="same")(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(320, (3, 3), padding="same", kernel_regularizer=None)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = concatenate([init, x])
    return m

def conv3_block(input, k=1, dropout=0.0, regularizer=None):
    init = input

    channel_axis = 1 if K.image_data_format() == "th" else -1

    # Check if input number of filters is same as 64 * k, else create Conv2D for this input
    if K.image_data_format() == "th":
        if init._keras_shape[1] != 64 * k:
            init = Conv2D(640, (1, 1), activation='linear', padding='same')(init)
    else:
        if init._keras_shape[-1] != 64 * k:
            init = Conv2D(640, (1, 1), activation='linear', padding='same')(init)

    x = Conv2D(640, (3, 3), padding='same')(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = Conv2D(640, (3, 3), padding='same', W_regularizer=regularizer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    m = concatenate([init, x])
    return m

def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1, wmark_regularizer=None, target_blk_num=1):
    """
    Creates a Wide Residual Network with specified parameters

    :param input: Input Keras object
    :param nb_classes: Number of output classes
    :param N: Depth of the network. Compute N = (n - 4) / 6.
              Example : For a depth of 16, n = 16, N = (16 - 4) / 6 = 2
              Example2: For a depth of 28, n = 28, N = (28 - 4) / 6 = 4
              Example3: For a depth of 40, n = 40, N = (40 - 4) / 6 = 6
    :param k: Width of the network.
    :param dropout: Adds dropout if value is greater than 0.0
    :param verbose: Debug info to describe created WRN
    :return:
    """
    def get_regularizer(blk_num, idx):
        if wmark_regularizer != None and target_blk_num == blk_num and idx == 0:
            print('target regularizer({}, {})'.format(blk_num, idx))
            return wmark_regularizer
        else:
            return None

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    for i in range(N):
        x = conv1_block(x, k, dropout, get_regularizer(1, i))
        nb_conv += 2

    x = MaxPooling2D((2,2), padding='same')(x)

    for i in range(N):
        x = conv2_block(x, k, dropout, get_regularizer(2, i))
        nb_conv += 2

    x = MaxPooling2D((2,2), padding='same')(x)

    for i in range(N):
        x = conv3_block(x, k, dropout, get_regularizer(3, i))
        nb_conv += 2

    x = AveragePooling2D((8,8), padding='same')(x)
    x = Flatten()(x)

    x = Dense(nb_classes, activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model

if __name__ == "__main__":
    #from keras.utils.visualize_util import plot
    from keras.layers import Input
    from keras.models import Model

    init = (3, 32, 32)

    wrn_28_10 = create_wide_residual_network(init, nb_classes=100, N=4, k=10, dropout=0.25)

    wrn_28_10.summary()
    #plot(wrn_28_10, "WRN-28-10.png", show_shapes=True, show_layer_names=True)
