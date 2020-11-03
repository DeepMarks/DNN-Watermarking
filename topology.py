 # this program defines the topology parameters
 # for the main black-box setting

from __future__ import division
from __future__ import print_function
import keras.utils.np_utils as kutils
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
import keras.backend as K
import numpy as np


def create_model(num_classes=10):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    # model.summary()

    return model
