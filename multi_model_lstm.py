import numpy as np
import gc

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Lambda, GaussianNoise
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers import Concatenate
from keras.layers import Dropout
from keras import regularizers

from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras import backend as K

import tensorflow as tf

from keras.utils import multi_gpu_model, plot_model

def multi_kernel_with_lstm(sample_pic_length, eye_movement, nclasses, lstm_output=False):

    inputs = Input(shape=(sample_pic_length, eye_movement, 4))  # 350 900->300, 4

    x1 = Conv2D(32, (1, 5), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x1_dropout = Dropout(0.7)(x1)
    average_pool1 = AveragePooling2D((1, 5), strides=(1, 5))(x1_dropout)
    x2 = Conv2D(32, (1, 10), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x2_dropout = Dropout(0.7)(x2)
    average_pool2 = AveragePooling2D((1, 5), strides=(1, 5))(x2_dropout)
    x3 = Conv2D(32, (1, 20), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x3_dropout = Dropout(0.7)(x3)
    average_pool3 = AveragePooling2D((1, 5), strides=(1, 5))(x3_dropout)
    x4 = Conv2D(32, (1, 30), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x4_dropout = Dropout(0.7)(x4)
    average_pool4 = AveragePooling2D((1, 5), strides=(1, 5))(x4_dropout)

    append_all = Concatenate(axis=-1)([average_pool1, average_pool2, average_pool3, average_pool4])

    conv_channel1 = Conv2D(16, (1, 1), kernel_regularizer=regularizers.l1(0.), activation='relu')(append_all)
    conv_dropout1 = Dropout(0.7)(conv_channel1)
    conv_channel2 = Conv2D(1, (1, 1), kernel_regularizer=regularizers.l1(0.), activation='relu')(conv_dropout1)
    conv_dropout2 = Dropout(0.7)(conv_channel2)

    Time_flatten = TimeDistributed(Flatten())(conv_dropout2)

    lstm_module = LSTM(128, activation="sigmoid", dropout=0.25, return_sequences=lstm_output)(Time_flatten)

    fc1 = Dense(32, activation='relu')(lstm_module)
    fc1_dropout = Dropout(0.5)(fc1)
    fc2 = Dense(nclasses, activation='softmax')(fc1_dropout)

    model = Model(inputs=inputs, outputs=fc2)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    plot_model(model, to_file='multi_kernel_lstm.png')
    print(model.summary())
    return model

def multi_kernel_with_lstm_gpu(sample_pic_length, eye_movement, nclasses, lstm_output=False):

    inputs = Input(shape=(sample_pic_length, eye_movement, 4))  # 350 900->300, 4

    x1 = Conv2D(32, (1, 5), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x1_dropout = Dropout(0.7)(x1)
    average_pool1 = AveragePooling2D((1, 5), strides=(1, 5))(x1_dropout)
    x2 = Conv2D(32, (1, 10), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x2_dropout = Dropout(0.7)(x2)
    average_pool2 = AveragePooling2D((1, 5), strides=(1, 5))(x2_dropout)
    x3 = Conv2D(32, (1, 20), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x3_dropout = Dropout(0.7)(x3)
    average_pool3 = AveragePooling2D((1, 5), strides=(1, 5))(x3_dropout)
    x4 = Conv2D(32, (1, 30), padding='same', kernel_regularizer=regularizers.l1(0.), activation='relu')(inputs)
    x4_dropout = Dropout(0.7)(x4)
    average_pool4 = AveragePooling2D((1, 5), strides=(1, 5))(x4_dropout)

    append_all = Concatenate(axis=-1)([average_pool1, average_pool2, average_pool3, average_pool4])

    conv_channel1 = Conv2D(16, (1, 1), kernel_regularizer=regularizers.l1(0.), activation='relu')(append_all)
    conv_dropout1 = Dropout(0.7)(conv_channel1)
    conv_channel2 = Conv2D(1, (1, 1), kernel_regularizer=regularizers.l1(0.), activation='relu')(conv_dropout1)
    conv_dropout2 = Dropout(0.7)(conv_channel2)

    Time_flatten = TimeDistributed(Flatten())(conv_dropout2)

    lstm_module = LSTM(128, activation="sigmoid", dropout=0.25, return_sequences=lstm_output)(Time_flatten)

    fc1 = Dense(32, activation='relu')(lstm_module)
    fc1_dropout = Dropout(0.7)(fc1)
    fc2 = Dense(nclasses, activation='softmax')(fc1_dropout)

    model = Model(inputs=inputs, outputs=fc2)
    plot_model(model, to_file='multi_kernel_lstm.png')

    print(model.summary())
    return model





# create_raw_model2(64, 2)
# multi_kernel_no_lstm(350, 300, 2)
# multi_kernel_with_lstm(350, 300, 2)
