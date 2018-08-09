from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras.initializers import glorot_uniform

def my_net(input_len = (56,), output_len = 61):
    
    X_input = Input(input_len)
    X = Dense(1000, activation='relu', name='fc1', kernel_initializer = glorot_uniform())(X_input)
    X = BatchNormalization(name = 'bn1')(X)
    
    X = Dense(1000, activation='relu', name='fc2', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(name = 'bn2')(X)
    
    X = Dense(1000, activation='relu', name='fc3', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(name = 'bn3')(X)
    
    X = Dropout(0.7)(X)
    
    X = Dense(output_len, activation='softmax', name='output', kernel_initializer = glorot_uniform())(X)

    model = Model(inputs = X_input, outputs = X, name='DemoNet')
    
    return model

all_x = np.loadtxt(open("data/X_train_1.csv","rb"),delimiter=",") 
all_y = np.loadtxt(open("data/Y_train_1.csv","rb"),delimiter=",")

all_x = all_x - np.mean(all_x, axis = 0)

split_num = int(all_x.shape[0]*0.7)

X_train = all_x[:split_num]
Y_train = all_y[:split_num]

X_test = all_x[split_num:]
Y_test = all_y[split_num:]

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape )  


model = my_net(input_len = (56,), output_len = 61)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs = 10, batch_size = 32, verbose = 1)

preds = model.evaluate(X_test, Y_test, verbose=1)
print ("Test Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

model.save("model/my_model.h5")