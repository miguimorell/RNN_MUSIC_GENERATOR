import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import music21 as m21
from RNN_MUSIC_GENERATOR.Generator.user_input_processing import translate_input
from RNN_MUSIC_GENERATOR.Generator.mapping_seed import map_seed
import os
DATASET_PATH = os.environ.get("DATASET_PATH")
ENCODED_PATH = os.environ.get("ENCODED_PATH")
ENCODED_PATH_INT = os.environ.get("ENCODED_PATH_INT")
#MAPPING_PATH = os.environ.get("MAPPING_PATH")
MAPPING_PATH="/home/miguimorell/code/miguimorell/RNN_MUSIC_GENERATOR/RNN_MUSIC_GENERATOR/Generator/mapping.json"
SEQUENCE_LENGTH = 32



class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="/home/miguimorell/code/miguimorell/RNN_MUSIC_GENERATOR/model.v2"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as f:
            self._mappings = json.load(f)

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
        seed_mapped=map_seed(seed,MAPPING_PATH)


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




    #CHECK
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


    def save_melody(self, melody, step_duration=0.25,format='midi', file_name= 'test.mid'):
        #create a music 21 stream
        stream= m21.stream.Stream()

        #parse all the symbols in the melody and create note/rest objects
        #e.g. 60_ _ _ r_ 62_
        start_symbol= None
        step_counter= 1

        for i, symbol in  enumerate(melody):
            #handle case in which we have a note/rest.
            if symbol != "_":
                pass
                #ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration= step_duration * step_counter #0.25*4 = 1
                    #handle rest
                    if start_symbol== 'r':
                        m21_event= m21.note.Rest(quarterLenth= quarter_length_duration)
                    #handle note
                    else:
                        m21_event= m21.note.Note(int(start_symbol),quarterLenth=quarter_length_duration)

                    stream.append(m21_event)

                #reset the step_counter
                    step_counter= 1

                start_symbol = symbol

            #handle case in which we have a prolongation sign "_"
            else:
                step_counter+= 1

        #write the m21 stream to a midifile
        stream.write(format, file_name)

        return stream



if __name__ == "__main__":
    mg = MelodyGenerator()
    user_input=[[True, False, False, False, False, False, False,False,True, False, False, False, False, False, False,False,True, False, False, False, False, False, False,False,True, False, False, False, False, False, False,False],
                [False, False, False, False, True,False, False, False,False, False, False, False, True,False, False, False,False, False, False, False, True,False, False, False,False, False, False, False, True,False, False, False],
                [True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,False]]


    melody = mg.generate_melody(user_input, 0.7)
    print(melody)
    midi=mg.save_melody(melody)




    #Reproducir
    # Crear un reproductor de transmisión (StreamPlayer)
    #reproductor = m21.midi.realtime.StreamPlayer(midi)

    # Reproducir el archivo MIDI
    #reproductor.play()
