 # the DeepMarks program defines the
 # functionality of the main black-box setting

from __future__ import division
from __future__ import print_function
import keras.utils.np_utils as kutils, keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import keras.backend as K, numpy as np
from keras.datasets import mnist
from scipy.special import comb

def compute_mismatch_threshold(C=10, Kp=50, p=0.05):
    prob_sum = 0
    p_err = 1 - 1.0 / C
    for i in range(Kp):
        cur_prob = comb(Kp, i, exact=False) * np.power(p_err, i) * np.power(1 - p_err, Kp - i)
        prob_sum = prob_sum + cur_prob
        if prob_sum > p:
            theta = i
            break

    return theta


def key_generation(x_train, y_train, marked_model, desired_key_len, num_classes=10, embed_epoch=20, modulation_strength=60000):
    key_len = np.dot(40, desired_key_len)
    batch_size = 128
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = kutils.to_categorical(y_train, num_classes)
    y_test = kutils.to_categorical(y_test, num_classes)
    key_gen_flag = 1
    while key_gen_flag:
        np.random.seed()
        x_retrain_rand = np.random.randint(256, size=(key_len, 784))
        x_retrain_rand = x_retrain_rand / 255.0
        np.random.seed()
        y_retrain_rand_vec = np.random.randint(10, size=(key_len, 1))
        y_retrain_rand = kutils.to_categorical(y_retrain_rand_vec, num_classes)
        x_train_subset = x_train[0:modulation_strength, :]
        y_train_subset = y_train[0:modulation_strength, :]
        x_retrain = np.vstack((x_train_subset, x_retrain_rand))
        y_retrain = np.vstack((y_train_subset, y_retrain_rand))
        unmarked_score = marked_model.evaluate(x_test, y_test, verbose=0)
        prediction_random_key = marked_model.predict(x_retrain_rand, batch_size=batch_size)
        preds = np.argmax(prediction_random_key, axis=1)
        preds = np.reshape(preds, (key_len, 1))
        mismatched_result = (preds != y_retrain_rand_vec) * 1
        random_unmarkMismatched_idx = np.argwhere(mismatched_result)
        random_unmarkMismatched_idx = random_unmarkMismatched_idx[:, 0]
        history = marked_model.fit(x_retrain, y_retrain, batch_size=batch_size, epochs=embed_epoch, shuffle=True, verbose=1, validation_data=(x_test, y_test))
        score = marked_model.evaluate(x_test, y_test, verbose=0)
        score = marked_model.evaluate(x_retrain_rand, y_retrain_rand, verbose=0)
        Perr_marked = 1 - score[1]
        mark_NN_err = int(Perr_marked * key_len)
        prediction_random_key = marked_model.predict(x_retrain_rand, batch_size=batch_size)
        preds = np.argmax(prediction_random_key, axis=1)
        preds = np.reshape(preds, (key_len, 1))
        matched_result = (preds == y_retrain_rand_vec) * 1
        matched_result = np.reshape(matched_result, (matched_result.shape[0], 1))
        random_MarkMatched_idx = np.argwhere(matched_result)
        random_MarkMatched_idx = random_MarkMatched_idx[:, 0]
        selected_key_idx = np.intersect1d(random_MarkMatched_idx, random_unmarkMismatched_idx)
        selected_keys = x_retrain_rand[np.array(selected_key_idx).astype(int), :]
        selected_keys_labels = y_retrain_rand[np.array(selected_key_idx).astype(int)]
        usable_key_len = selected_keys.shape[0]
        print('Usable key len is: ', usable_key_len)
        if usable_key_len < desired_key_len:
            key_gen_flag = 1
            print(' Desired key length is {}, Longer key needed, skip this test. '.format(desired_key_len))
        else:
            key_gen_flag = 0
            selected_keys = selected_keys[0:desired_key_len, :]
            selected_keys_labels = selected_keys_labels[0:desired_key_len]
            np.save('result/keyRandomImage' + '_keyLength' + str(desired_key_len) + '.npy', selected_keys)
            np.savetxt('result/keyRandomLabel' + '_keyLength' + str() + '.txt', selected_keys_labels, fmt='%i', delimiter=',')
            actual_key_len = selected_keys.shape[0]
            marked_model.save_weights('result/markedWeights' + '.h5')
            print('WM key generation finished. Save watermarked model. ')
            break

    return (
     selected_keys, selected_keys_labels)


def count_response_mismatch(Y_preds, Y_key):
    num_mismatch = np.sum((Y_preds == Y_key) * 1)
    return num_mismatch
