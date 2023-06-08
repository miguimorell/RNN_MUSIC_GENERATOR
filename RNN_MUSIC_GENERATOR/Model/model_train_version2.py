import tensorflow
import os
from tensorflow import keras
from keras.layers import  LSTM, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from RNN_MUSIC_GENERATOR.Processing_NEW.processing import process_data, SEQUENCE_LENGTH

OUTPUT_UNITS = 54 #change for ours
NUM_UNITS = [256] #check if is not too much??!!
LOSS = "sparse_categorical_crossentropy" #check
LEARNING_RATE = 0.001 #check if should do a GridSearch
EPOCHS = 50 #check
BATCH_SIZE = 64 #Check
SAVE_MODEL_PATH = "model.a1" #change if we want different name


def init_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model)
    """

    # create the model architecture
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # compile model
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    return model



def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """Train and save TF model.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """
    # generate the training sequences
    X_train, y_train = process_data()

    # build the network
    model = init_model(output_units, num_units, loss, learning_rate)
    model.summary()

    # train the model
    es = EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[es])


    # save the model
    model.save(SAVE_MODEL_PATH)

    return model


if __name__ == "__main__":
    train()
