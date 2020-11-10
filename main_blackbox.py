 # this project combines the ideas behind the watermarking
 # frameworks of Rouhani et al. (2019) and Quan et al. (2020).
 
 -------------------------------------------------------------
 
 # main program to watermark a DNN and
 # detect original ownership in a black-box setting

import DeepMarks
from DeepMarks import key_generation
from DeepMarks import count_response_mismatch
from DeepMarks import compute_mismatch_threshold

import keras.utils.np_utils as kutils
from topology import create_model
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

# from blackboxWM_mnistmlp import blackboxWM_demo

if __name__ == '__main__':
    
    num_classes = 10
    batch_size = 128

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = kutils.to_categorical(y_train, num_classes)
    y_test = kutils.to_categorical(y_test, num_classes)


    key_len = 20              ## desired WM key length
    embed_lr = 0.0008
    p_threshold = 0.0001
    embed_epoch = 2 

    ## ---- Embed WM ------ ##
    model = create_model()
    model.load_weights('result/unmarked_weights.h5')
    model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=embed_lr, momentum=0.9, decay=0.0, nesterov=True), metrics=['accuracy'])
    X_key, Y_key = key_generation(x_train, y_train, model, key_len, num_classes, embed_epoch)


    ## ----- Detect WM ------ ##
    marked_model = create_model()
    marked_model.load_weights('result/markedWeights'+'.h5')
    marked_model.compile(loss='categorical_crossentropy',
                optimizer=SGD(lr=embed_lr, momentum=0.9, decay=0.0, nesterov=True), metrics=['accuracy'])
    preds_onehot = marked_model.predict(X_key, batch_size = batch_size )
    Y_preds = np.reshape(np.argmax(preds_onehot, axis=1), (key_len, 1))
    m = count_response_mismatch(Y_preds, Y_key)
    theta = compute_mismatch_threshold(C=num_classes, Kp=key_len, p=p_threshold) # pk = 1/C, |K|: # trials

    print('Probability threshold p is ', p_threshold)
    print('Mismatch threshold is : ', theta)
    print('Mismatch count of marked model on WM key set = ', m)
    print("If the marked model is correctly authenticated by owner: ", m < theta)
