import json
import numpy as np
import tensorflow as tf
import os
import json
from tensorflow import keras
import music21 as m21
from RNN_MUSIC_GENERATOR.Generator.user_input_processing import translate_input
from RNN_MUSIC_GENERATOR.Generator.mapping_seed import map_seed
from RNN_MUSIC_GENERATOR.registry import load_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
DATASET_PATH = os.environ.get("DATASET_PATH")
ENCODED_PATH = os.environ.get("ENCODED_PATH")
ENCODED_PATH_INT = os.environ.get("ENCODED_PATH_INT")
#MAPPING_PATH = os.environ.get("MAPPING_PATH")
MAPPING = {"r": 0,
    "59": 1,
    "61": 2,
    "70": 3,
    "38": 4,
    "34": 5,
    "39": 6,
    "46": 7,
    "53": 8,
    "36": 9,
    "63": 10,
    "29": 11,
    "44": 12,
    "/": 13,
    "30": 14,
    "65": 15,
    "69": 16,
    "47": 17,
    "62": 18,
    "48": 19,
    "43": 20,
    "68": 21,
    "54": 22,
    "50": 23,
    "31": 24,
    "52": 25,
    "37": 26,
    "51": 27,
    "40": 28,
    "_": 29,
    "41": 30,
    "42": 31,
    "32": 32,
    "49": 33,
    "66": 34,
    "45": 35,
    "67": 36,
    "57": 37,
    "64": 38,
    "35": 39,
    "33": 40,
    "58": 41,
    "60": 42,
    "56": 43
}


SEQUENCE_LENGTH = 32
SAVE_MODEL_PATH = os.environ.get("SAVE_MODEL_PATH")


class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, app, model_origin='mlflow'):
        """Constructor that initialises TensorFlow mlflow model"""
        #app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allows all origins
            allow_credentials=True,
            allow_methods=["*"],  # Allows all methods
            allow_headers=["*"],  # Allows all headers
)
        app.state.model = load_model(model_origin='mlflow')
        #app.state.model = keras.models.load_model(SAVE_MODEL_PATH)
        self.model = app.state.model
        self._mappings = MAPPING
        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, user_input, temperature):
        """Generates a melody using the DL model and returns a midi file.

        :param seed (str): Melody seed with the notation used to encode the dataset
        :param num_steps (int): Number of steps to be generated
        :param max_sequence_len (int): Max number of steps in seed to be considered for generation
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return melody (list of str): List with symbols representing a melody
        """

        #Change from FALSE/TRUE to r,_,45,_
        seed=translate_input(user_input)

        #Mapping seed
        seed_mapped=map_seed(seed,self._mappings)


        #OneHotEncode AND RESHAPE seeds
        #OHE seeds
        seed_ohe=[[],[],[]]
        for index,instrument in enumerate(seed_mapped):
            for element in instrument:
                seed_ohe[index].append(keras.utils.to_categorical(element,num_classes=len(self._mappings), dtype="int32"))
        seed_ohe=np.array([seed_ohe])

        #RESHAPE
        seed_ohe=np.transpose(seed_ohe,(0,2,1,3))
        d0,d1,d2,d3=seed_ohe.shape
        seed_ohe=np.reshape(seed_ohe,(d0,d1,d2*d3))

        #make prediction::
        print("Predicting:")
        prediction=self.model.predict(seed_ohe)


        #receive answer and translate
        #Find biggest probability value
        max_indices= np.argmax(prediction[0],axis=1)
        #Map prediction to values
        for index in range(0,32):
            print(max_indices[index])

        outputs=[]
        for observation in max_indices:
            output_symbol = [k for k, v in self._mappings.items() if v == observation]
            outputs.append(output_symbol[0])

        return outputs


    #CHECK MIGUEL
    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        :param predictions (nd.array): Array containing probabilities for each of the possible outputs.
        :param temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
            A number closer to 1 makes the generation more unpredictable.

        :return index (int): Selected output symbol
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index

@app.get("/predict")
def predict(CH,CK,SN):
    """
    Make a single course prediction.
    Receives X, a list of lists from the instruments, pass to the MelodyGenerator class the input,
    and returns the prediction.
    """
    CH_list = CH.split(",")
    # Convert the string values to their corresponding boolean values
    # print("CH_LIST")
    # print(CH_list)
    CH_list = [value == "True" for value in CH_list]
    # print("CH_LIST_BOOLEAN")
    # print(CH_list)

    CK_list = CK.split(",")
    # Convert the string values to their corresponding boolean values
    CK_list = [value == "True" for value in CK_list]

    SN_list = SN.split(",")
    # Convert the string values to their corresponding boolean values
    SN_list = [value == "True" for value in SN_list]

    X = [CH_list,CK_list,SN_list]


    mg = MelodyGenerator(app= FastAPI(), model_origin='mlflow')
    #X= [[True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False],
    #            [True, False, False, False, False, False, False,False,True, False, False, False, False, False, False,False,True, False, False, False, False, False, False,False,True, False, False, False, False, False, False,False],
    #            [False, False, False, False, True,False, False, False,False, False, False, False, True,False, False, False,False, False, False, False, True,False, False, False,False, False, False, False, True,False, False, False]]
    melody = mg.generate_melody(X, 0.7)
    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api sresponse to JSON
    my_dict = {index: value for index, value in enumerate(melody)}
    return my_dict


@app.get("/")
def root():
    return {
    'greeting': 'Hello to RNN Music Generator'
}


#http://127.0.0.1:8000/predict?CH=True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False&CK=True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False&SN=False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,True,False,False,False
#TESTING API
