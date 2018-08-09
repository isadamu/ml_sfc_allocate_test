from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Conv2D, Flatten, ZeroPadding2D, MaxPooling2D
from keras.models import Model
from keras.initializers import glorot_uniform

def my_net(input_len = (5,5,12), classes = 6 ):
    
    X_input = Input(input_len)
    
    # padding to 6 * 6
    X = ZeroPadding2D((1, 1))(X_input)
    
    # first conv
    X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = 'conv1', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    # second conv
    X = Conv2D(filters = 128, kernel_size = (3, 3), strides = (1,1), padding = 'same', name = 'conv2', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    # Pooling
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    
    # output layer
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='fc1', kernel_initializer = glorot_uniform())(X)
    X = Dropout(0.5)(X)
    X = Dense(classes, activation='softmax', name='fc2', kernel_initializer = glorot_uniform())(X)

    model = Model(inputs = X_input, outputs = X, name='SFC_Net')
    
    return model

X_train = np.load("data/X_train_new_2.npy")
Y_train = np.load("data/Y_train_new_2.npy")

Net = {}

for i in range(5):
    print("##############################################")
    print("i = " + str(i))
    net = my_net(input_len = (5,5,12), classes = 6 )
    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    net.fit(X_train, Y_train[:,i,:], epochs = 100, batch_size = 32, verbose = 1)
    Net["net" + str(i)] = net

for i in range(5):
    net = Net["net" + str(i)]
    net.save("model_advance/net" + str(i) + ".h5")

# TEST
X_test = np.load("data/X_test_new.npy")
Y_test = np.load("data/Y_test_new.npy")

unlike = np.zeros(Y_test.shape[0])

for i in range(5):
    net = Net["net" + str(i)]
    pred = np.argmax(net.predict(X_test), axis=1)
    real = np.argmax(Y_test[:,i,:], axis=1)
    unlike[pred != real] = 1

print("accuracy = ")
print(1 - (np.sum(unlike)/len(unlike)))
