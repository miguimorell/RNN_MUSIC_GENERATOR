import os

from RNN_MUSIC_GENERATOR.Processing_NEW.load_songs import load_songs
from RNN_MUSIC_GENERATOR.Processing_NEW.create_encoded_files import create_encoded_files
from RNN_MUSIC_GENERATOR.Processing_NEW.create_master_file import create_master_file
from RNN_MUSIC_GENERATOR.Processing_NEW.load_encoded_songs import load_encoded_songs
from RNN_MUSIC_GENERATOR.Processing_NEW.create_mapping import create_mapping
from RNN_MUSIC_GENERATOR.Processing_NEW.map_file import map_file
from RNN_MUSIC_GENERATOR.Processing_NEW.generate_training_data import generate_training_sequences
from RNN_MUSIC_GENERATOR.Processing_NEW.convert_dictionary import convert_dictionary

'''
MIDI FILES GIVES ME THE FOLLOWING INFORMATION:
NOTA:C4 , DURATION: 0.25s, VELOCIDAD: 125
SILENCIO: , DURATION:
'''
DATASET_PATH = os.environ.get("DATASET_PATH")
ENCODED_PATH = os.environ.get("ENCODED_PATH")
ENCODED_PATH_INT = os.environ.get("ENCODED_PATH_INT")
MAPPING_PATH = os.environ.get("MAPPING_PATH")
SEQUENCE_LENGTH = 32


def process_data():

    songs,files_name = load_songs(DATASET_PATH)

    #generate encoded files
    create_encoded_files(songs,files_name,ENCODED_PATH)

    #create master file: 1 line BS, 2 line velocity,3 line KICK, 4 line velocity, etc. TOTAL 8 lines
    create_master_file(SEQUENCE_LENGTH,ENCODED_PATH)

    #load songs from master file
    songs = load_encoded_songs(ENCODED_PATH,'Master_File',0)

    #create file mapping with vocabulary to then convert characters to integers
    if not os.path.exists(MAPPING_PATH):
        create_mapping(songs,MAPPING_PATH)

    #we map the master_file according to the vocabulary generated in the previous step
    map_file(songs,MAPPING_PATH,ENCODED_PATH_INT)

    #load songs from master file int
    mapped_songs = load_encoded_songs(ENCODED_PATH_INT,'Master_File_int',1)
    #get rid of spaces
    #mapped_songs = [element.replace(" ", "").strip() for element in mapped_songs]
    #convert everything to int
    #mapped_songs = [[int(element.replace(" ", "").strip()) for element in sublist] for sublist in mapped_songs]


    print('-----')
    print(len(mapped_songs[0]))
    print(len(mapped_songs[1]))
    print(len(mapped_songs[2]))
    print(len(mapped_songs[3]))
    print(len(mapped_songs[4]))
    print(len(mapped_songs[5]))
    print(len(mapped_songs[6]))
    print(len(mapped_songs[7]))
    print('-----')

    #this return a dictionary, with each key being a sequence, for both input and target
    inputs,targets = generate_training_sequences(mapped_songs,SEQUENCE_LENGTH, MAPPING_PATH)

    print('KEYS')
    print(len(inputs))
    print('SECUENCIA 0')
    print(inputs[0])
    print(inputs[0].shape)
    print('SECUENCIA 0, OBSERVACION 0')
    print(inputs[0][0])
    print(inputs[0][0].shape)
    X_train = convert_dictionary(inputs)
    #y_train = convert_dictionary(targets)

    #return X_train,y_train

    #np.set_printoptions(threshold=np.inf)
    #save_path = os.path.join(ENCODED_PATH_INT, 'FEATURES.txt')
    #with open(save_path, "w") as fp:
    #    for key,values in inputs.items():
    #        fp.write(str(values))
    #        print(str(values))
    #    fp.write('\n')

    #save_path = os.path.join(ENCODED_PATH_INT, 'TARGETS')
    #with open(save_path, "w") as fp:
    #    for key,values in targets.items():
    #        fp.write(str(values))
    #        fp.write('\n')

if __name__ == "__main__":
    process_data()
