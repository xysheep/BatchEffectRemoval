'''
Created on Dec 5, 2016

@author: urishaham
'''
import sys
import os
sys.path.append(os.path.abspath('./src'))
import keras.optimizers
from keras.layers import Input, Dense, merge, Activation, add
from keras.models import Model
from keras import callbacks as cb
import numpy as np
from keras.layers.normalization import BatchNormalization
import CostFunctions as cf
import Monitoring as mn
from keras.regularizers import l2
from sklearn import decomposition
from keras.callbacks import LearningRateScheduler
import math
from keras import initializers
from numpy import genfromtxt
import sklearn.preprocessing as prep
import tensorflow as tf
import keras.backend as K

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(9464)
source = np.random.multivariate_normal([1, 1], [[1, 0], [0, 1]], 100)
source = np.concatenate(
    (source, np.random.multivariate_normal([1, 6], [[1, 0], [0, 1]], 100)))
source = np.concatenate(
    (source, np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 100)))
source = np.concatenate(
    (source, np.random.multivariate_normal([6, 1], [[1, 0], [0, 1]], 100)))


target = np.random.multivariate_normal([-1, -1], [[1, 0], [0, 1]], 100)
target = np.concatenate(
    (target, np.random.multivariate_normal([-1, 8], [[1, 0], [0, 1]], 100)))
target = np.concatenate(
    (target, np.random.multivariate_normal([8, 8], [[1, 0], [0, 1]], 100)))
target = np.concatenate(
    (target, np.random.multivariate_normal([8, -1], [[1, 0], [0, 1]], 100)))


preprocessor = prep.StandardScaler().fit(source)
source = preprocessor.transform(source)
target = preprocessor.transform(target)

plt.figure(9464)
plt.subplot(1, 2, 1)
plt.scatter(source[:, 0], source[:, 1], color='r')
plt.scatter(target[:, 0], target[:, 1], color='g')
#############################
######## train MMD net ######
#############################
mmdNetLayerSizes = [25, 25]
l2_penalty = 1e-2
inputDim = target.shape[1]
calibInput = Input(shape=(inputDim,))
block1_bn1 = BatchNormalization()(calibInput)
block1_a1 = Activation('relu')(block1_bn1)
block1_w1 = Dense(mmdNetLayerSizes[0], activation='linear', kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a1)
block1_bn2 = BatchNormalization()(block1_w1)
block1_a2 = Activation('relu')(block1_bn2)
block1_w2 = Dense(inputDim, activation='linear', kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block1_a2)
block1_output = add([block1_w2, calibInput])
block2_bn1 = BatchNormalization()(block1_output)
block2_a1 = Activation('relu')(block2_bn1)
block2_w1 = Dense(mmdNetLayerSizes[1], activation='linear', kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a1)
block2_bn2 = BatchNormalization()(block2_w1)
block2_a2 = Activation('relu')(block2_bn2)
block2_w2 = Dense(inputDim, activation='linear', kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block2_a2)
block2_output = add([block2_w2, block1_output])
block3_bn1 = BatchNormalization()(block2_output)
block3_a1 = Activation('relu')(block3_bn1)
block3_w1 = Dense(mmdNetLayerSizes[1], activation='linear', kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a1)
block3_bn2 = BatchNormalization()(block3_w1)
block3_a2 = Activation('relu')(block3_bn2)
block3_w2 = Dense(inputDim, activation='linear', kernel_regularizer=l2(l2_penalty),
                  kernel_initializer=initializers.RandomNormal(stddev=1e-4))(block3_a2)
block3_output = add([block3_w2, block2_output])

calibMMDNet = Model(inputs=calibInput, outputs=block3_output)

# learning rate schedule


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 150.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


lrate = LearningRateScheduler(step_decay)
# train MMD net
optimizer = keras.optimizers.rmsprop(lr=0.0)

np.random.seed(9464)

calibMMDNet.compile(optimizer=optimizer, loss=lambda y_true, y_pred:
                    cf.MMD(block3_output, target).KerasCost(y_true, y_pred))
K.get_session().run(tf.global_variables_initializer())
sourceLabels = np.zeros(source.shape[0])

calibMMDNet.fit(source, sourceLabels, epochs=150,
                batch_size=90, verbose=1, validation_split=0.1,
                callbacks=[lrate, mn.monitorMMD(source, target, calibMMDNet.predict),
                           cb.EarlyStopping(monitor='val_loss', patience=50, mode='auto')])

##############################
###### evaluate results ######
##############################

calibratedSource = calibMMDNet.predict(source)
plt.figure(9464)
plt.subplot(1, 2, 2)
plt.scatter(calibratedSource[:, 0], calibratedSource[:, 1], color='r')
plt.scatter(target[:, 0], target[:, 1], color='g')
plt.show(block=True)

'''
# save models
autoencoder.save(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_DAE.h5'))                 
calibMMDNet.save_weights(os.path.join(io.DeepLearningRoot(),'savedModels/person1_baseline_ResNet_weights.h5'))  
'''
