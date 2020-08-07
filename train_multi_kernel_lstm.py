import numpy as np
import os
import random
import scipy.io
import multi_model
import keras

from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras.callbacks import Callback
from keras.utils import multi_gpu_model


MODEL_PATH = '/home/amax/DL/sun/asd/multi_kernel_lstm_model/'
TOTAL_PERSON = 39
TRAIN_PERSON_NUMBER = 33
TEST_PERSON_NUMBER = 6
LAL_PERSON_NUMBER = 2

nclasses = 2
sample_num = 400
pic_per_sample = 350
eye_movement_length = 300
nchann = 4
epoch = 200

# use multi gpu training
os.environ["CUDA_VISIBLE_DEVICES"]='4,5,6'
config = tf.ConfigProto()

# use all gpus available (defined by: CUDA_VISIBLE_DEVICES)
def get_available_gpus():
    from tensorflow.python.client import device_lib as _device_lib 
    local_device_protos = _device_lib.list_local_devices()
    return len([x.name for x in local_device_protos if x.device_type == 'GPU'])

num_gpu = get_available_gpus()

# print the prediction
class CollectOutputAndTarget(Callback):
    def __init__(self):
        super(CollectOutputAndTarget, self).__init__()
        self.targets = []  #
        self.outputs = []  # collect y_pred batches

        # the shape of these 2 variables will change according to batch shape
        # to handle the "last batch", specify `validate_shape=False`
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_epoch_end(self, batch, logs=None):
        # evaluate the variables and save them into lists
        self.targets.append(K.eval(self.var_y_true))
        self.outputs.append(K.eval(self.var_y_pred))

# multi GPU save model
class CustomModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss


# load data
X = np.load('/home/amax/DL/sun/asd/train_data_noise400.npy')   # (13200, 350, 300, 4)  33
Y = np.load('/home/amax/DL/sun/asd/train_label_noise400.npy')  # (13200, 2)    
testX = np.load('/home/amax/DL/sun/asd/test_data_noise400.npy')     # (2400, 350, 300, 4)    6
testY = np.load('/home/amax/DL/sun/asd/test_label_noise400.npy')    # (2400, 2)
trainX = X[:31*400]
trainY = Y[:31*400]
valX = X[31*400:]    #  (1200, 350, 300, 4) 2
valY = Y[31*400:]    #  (1200, 2)   2
# shuffle validation data, shuffle data with same state
state = np.random.get_state()
np.random.shuffle(valX)
np.random.set_state(state)
np.random.shuffle(valY)


# training  http://www.manongjc.com/article/31408.html

model = multi_model.multi_kernel_with_lstm_gpu(pic_per_sample, eye_movement_length, nclasses)
mgpu_model = multi_gpu_model(model, gpus=num_gpu)
mgpu_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
cbk = CollectOutputAndTarget()

fetches = [tf.assign(cbk.var_y_pred, mgpu_model.outputs[0], validate_shape=False)]
mgpu_model._function_kwargs = {'fetches': fetches}

checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_PATH, 'model_multi_kernel_lstm-{epoch:02d}.h5'))
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

mgpu_model.fit(X, Y, batch_size=128, epochs=epoch, verbose=1, callbacks=[checkpoint, tensorboard], 
         shuffle=True, validation_data=(valX, valY))
mgpu_model.save('multi_kernel_lstm.h5')

# testing
score = mgpu_model.evaluate(testX, testY, verbose=1)
print('test accuracy:', score[1])


