from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38 #change for ours
NUM_UNITS = 256 #check if is not too much??!!
LOSS = "sparse_categorical_crossentropy" #check
LEARNING_RATE = 0.001 #check if should do a cross_validation
EPOCHS = 50 #check
BATCH_SIZE = 64 #Check
SAVE_MODEL_PATH = "model.h5" #change


def init_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model): Where the magic happens :D
    """

    # create the model architecture
    model = Model()
    model = Input(shape=(None, output_units))
    model.add(LSTM(units= num_units)) # activation='tanh' ?
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_units, activation='softmax'))

    # compile model
    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=learning_rate,beta_1=0.9),
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
    X_train, y_train = generate_training_sequences(SEQUENCE_LENGTH)

    # build the network
    model = init_model(output_units, num_units, loss, learning_rate)
    model.summary()

    # train the model
    es = EarlyStopping(patience=10, restore_best_weights=True)

    history= model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[es])


    # save the model
    final_model= history[1]
    final_model.save(SAVE_MODEL_PATH)

    return (model, history)


if __name__ == "__main__":
    train()
