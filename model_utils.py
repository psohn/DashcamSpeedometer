# importing libraries
import keras
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv3D, MaxPooling2D, MaxPooling3D
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import pandas as pd



def create_model(input_shape):
    '''
    this function creates the model for training
    '''
    
    # setup model
    model = Sequential()
        
    # convolution and pooling layer 1
    model.add(Conv2D(filters = 12,
                     kernel_size = (5, 5),
                     strides = (1, 1),
                     activation = 'elu',
                    input_shape = input_shape))
    
    model.add(MaxPooling2D(pool_size = (2, 2),
                           strides = (2, 2)))

    # convolution and pooling layer 2
    model.add(Conv2D(filters = 24,
                     kernel_size = (5, 5),
                     strides = (1, 1),
                     activation = 'elu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2),
                           strides = (2, 2)))

    # convolution and pooling layer 3
    model.add(Conv2D(filters = 36,
                     kernel_size = (3, 3),
                     strides = (1, 1),
                     activation = 'elu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2),
                           strides = (2, 2)))

    # convolution and pooling layer 4
    model.add(Conv2D(filters = 48,
                     kernel_size = (3, 3),
                     strides = (1, 1),
                     activation = 'elu'))
    
    model.add(MaxPooling2D(pool_size = (2, 2),
                           strides = (2, 2)))
     
    # flatten layer
    model.add(Flatten())
    
    # dense layer 1
    model.add(Dense(units = 1164,
                   activation = 'elu'))
    
    model.add(Dropout(rate = 0.5))
    
    # dense layer 2
    model.add(Dense(units = 50,
                   activation = 'elu'))
    
    model.add(Dropout(rate = 0.5))

    # dense layer 3
    model.add(Dense(units = 10,
                   activation = 'elu'))
    
    model.add(Dropout(rate = 0.5))
    
    # dense layer 4
    model.add(Dense(units = 1,
             activation = 'linear'))
    
    # compilation
    model.compile(Adam(lr = 0.001),
                 loss = 'mse',
                 metrics = ['mse'])
    
    return model



def initiate_model(x, y, n_epochs, validation):
    '''
    this function trains a model from scratch
    '''
    
    # initializing model
    print('---creating model---')
    input_shape = x.shape[1:]
    model = create_model(input_shape)
    print('---model created... now training---')
    
    # training model
    for i in range(n_epochs):
        
        print('---' + str(i + 1) + '/' + str(n_epochs) + ' epochs being trained---')
        model.fit(x, y, 
                  epochs = 1, 
                  verbose = 1, 
                  validation_data = validation, 
                  batch_size = 64)
        print('---saving model---')
        save_model(model, filepath = 'model/model')
    print('---model done training---')
    


def train_model(x, y, n_epochs, validation, model_path):
    '''
    this function continues training a model
    '''
    
    # initializing model
    print('---initializing model---')
    model = load_model(filepath = model_path)
    print('---model initialized... now training---')
    
    # training model
    for i in range(n_epochs):
        
        print('---' + str(i + 1) + '/' + str(n_epochs) + ' epochs being trained---')
        model.fit(x, y, 
                  epochs = 1, 
                  verbose = 1, 
                  validation_data = validation, 
                  batch_size = 64)
        print('---saving model---')
        save_model(model, filepath = 'model/model')
    print('---model done training---')