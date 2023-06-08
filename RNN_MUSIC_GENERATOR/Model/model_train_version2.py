import tensorflow
import os
from tensorflow import keras
from keras.layers import  LSTM, Dense, Dropout, SimpleRNN
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from RNN_MUSIC_GENERATOR.Processing_NEW.processing import process_data, SEQUENCE_LENGTH

#OUTPUT_UNITS = 2 #change for ours
#NUM_UNITS = 54 #check if is not too much??!!
#LOSS = "mean_squared_error"
LOSS = 'categorical_crossentropy'
LEARNING_RATE = 0.001 #check if should do a GridSearch
EPOCHS = 20 #check
BATCH_SIZE = 32 #Check
FEATURES = 6
#SAVE_MODEL_PATH = "model.a1" #change if we want different name
SAVE_MODEL_PATH = "model.a2" #change if we want different name

# def init_model(num_units, loss, learning_rate,shape_1,shape_2):
#     model = Sequential()
#     model.add(LSTM(units=int(num_units), activation='relu', return_sequences=True, input_shape=(shape_1, shape_2)))
#     model.add(LSTM(units=int(num_units/10), activation='relu', return_sequences=True))
#     model.add(Dense(2))

#     #compile model
#     model.compile(loss=loss,
#                   optimizer=Adam(learning_rate=learning_rate),
#                   metrics=["mae"])
#     return model

def init_model(num_units, loss, learning_rate,shape_1,shape_2):
    model = Sequential()
    model.add(LSTM(units=int(num_units), activation='relu', return_sequences=True, input_shape=(shape_1, shape_2)))
    model.add(LSTM(units=int(num_units/10), activation='relu', return_sequences=True))
    model.add(Dense(99,activation = 'softmax'))

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
    X_train, y_train,X_test,y_test = process_data()

    print('SHAPE X TRAIN')
    print(X_train.shape)
    print('SHAPE y TRAIN')
    print(y_train.shape)
    # build the network
    #shape = (X_train.shape[1],X_train.shape[2:])

    shape_1 = X_train.shape[1] # observations
    shape_2 = X_train.shape[2] # features
    model = init_model(shape_2, loss, learning_rate,shape_1,shape_2)
    model.summary()

    # train the model
    es = EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[es])


    # save the model
    model.save(SAVE_MODEL_PATH)

    return model,X_test,y_test


if __name__ == "__main__":

    model,X_test,y_test= train()

    #model = keras.models.load_model(SAVE_MODEL_PATH)
    #X_train, y_train,X_test,y_test = process_data()
    y_pred = model.predict(X_test)

    print('Y PREDICT')
    print(y_pred[0])

    print('Y TEST')
    print(y_test[0])
