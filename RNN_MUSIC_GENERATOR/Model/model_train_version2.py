import tensorflow
import os
from tensorflow import keras
from keras.layers import  LSTM, Dense, Dropout, SimpleRNN
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from RNN_MUSIC_GENERATOR.Processing_NEW.processing import process_data, SEQUENCE_LENGTH
import numpy as np
import json


MAPPING_PATH = os.environ.get("MAPPING_PATH")
LOSS = 'sparse_categorical_crossentropy'
LEARNING_RATE = 0.001 #check if should do a GridSearch
EPOCHS = 100 #check
BATCH_SIZE = 32 #Check
FEATURES = 6
#SAVE_MODEL_PATH = "model.a1" #regression
SAVE_MODEL_PATH = "model.v2" #first try with classification, no velocity

def init_model(num_units, loss, learning_rate,shape_1,shape_2,num_classes):
    model = Sequential()
    model.add(LSTM(units=int(num_units), activation='relu', return_sequences=True, input_shape=(shape_1, shape_2)))
    model.add(LSTM(units=int(num_units/10), activation='relu', return_sequences=True))
    model.add(Dense(64, activation='relu'))  # Reduce the number of units to match the desired output shape
    model.add(Dense(num_classes, activation='softmax'))  # Set the output shape to (32, 2)

    #compile model
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    return model

def train(loss=LOSS, learning_rate=LEARNING_RATE):
    """Train and save TF model.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """
    # generate the training sequences
    X_train, y_train,X_test,y_test,X_val,y_val = process_data()

    print('SHAPE X TRAIN')
    print(X_train.shape)
    print('SHAPE y TRAIN')
    print(y_train.shape)
    # build the network
    #shape = (X_train.shape[1],X_train.shape[2:])

    #because the mapping changes every time we code is run, the number of classes may change
    #if more were added when song were added. So we need the lenght of the dictionary
    #for the last layer

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    classes = len(mappings)
    shape_1 = X_train.shape[1] # observations
    shape_2 = X_train.shape[2] # features
    model = init_model(shape_2, loss, learning_rate,shape_1,shape_2,classes)
    model.summary()

    # train the model
    es = EarlyStopping(patience=10,restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=EPOCHS,
              validation_data=(X_val,y_val),
              batch_size=BATCH_SIZE,
              callbacks=[es])


    # save the model
    model.save(SAVE_MODEL_PATH)

    return model,X_test,y_test


if __name__ == "__main__":

    #if the model is not trained
    #model,X_test,y_test= train()

    #if the model is trained
    model = keras.models.load_model(SAVE_MODEL_PATH)
    X_train, y_train,X_test,y_test,X_val,y_val = process_data()

    #make prediction
    y_pred = model.predict(X_test)

    print('Y PREDICT TYPE')
    print(type(y_pred[0]))

    print('Y PREDICT LEN')
    print(y_pred.shape)

    print('Y TEST LEN')
    print(y_test.shape)

    print('Y PREDICT[0] LEN')
    print(y_pred[0].shape)

    print('Y TEST[0] LEN')
    print(y_test[0].shape)

    for i in range(0,10):
        # Get the indices of the maximum values for each observation
        max_indices = np.argmax(y_pred[i], axis=1)
        #max_values = y_pred[i, np.arange(y_pred.shape[1]), max_indices]  # Retrieve the max values using indexing

        print('Y PREDICT, Y TEST')
        for index in range(0,32):
            print (max_indices[index],y_test[i][index])
