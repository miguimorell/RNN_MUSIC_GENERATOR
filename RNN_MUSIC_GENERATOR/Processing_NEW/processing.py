import os

from RNN_MUSIC_GENERATOR.Processing_NEW.load_songs import load_songs
from RNN_MUSIC_GENERATOR.Processing_NEW.create_encoded_files import create_encoded_files
from RNN_MUSIC_GENERATOR.Processing_NEW.create_master_file import create_master_file
from RNN_MUSIC_GENERATOR.Processing_NEW.load_encoded_songs import load_encoded_songs
from RNN_MUSIC_GENERATOR.Processing_NEW.create_mapping import create_mapping
from RNN_MUSIC_GENERATOR.Processing_NEW.map_file import map_file
from RNN_MUSIC_GENERATOR.Processing_NEW.generate_training_data import generate_training_sequences
from RNN_MUSIC_GENERATOR.Processing_NEW.convert_dictionary import convert_dictionary_x,convert_dictionary_y

'''
MIDI FILES GIVES ME THE FOLLOWING INFORMATION:
NOTA:C4 , DURATION: 0.25s, VELOCIDAD: 125
SILENCIO: , DURATION:
'''
DATASET_PATH = os.environ.get("DATASET_PATH")
ENCODED_PATH = os.environ.get("ENCODED_PATH")
ENCODED_PATH_INT = os.environ.get("ENCODED_PATH_INT")
MAPPING_PATH = os.environ.get("MAPPING_PATH")
#SEQUENCE_LENGTH = 64 #for time_step = 0.125
SEQUENCE_LENGTH = 32 # for time_step = 0.25


def process_data():

    print('Loading songs...')
    songs,files_name = load_songs(DATASET_PATH)
    print('Finished Loading songs...')

    #generate encoded files
    print('Encoding...')
    create_encoded_files(songs,files_name,ENCODED_PATH)
    print('Finished Encoding...')
    #create master file: 1 line BS, 2 line velocity,3 line KICK, 4 line velocity, etc. TOTAL 8 lines
    print('Creating Master File...')
    create_master_file(SEQUENCE_LENGTH,ENCODED_PATH)
    print('Finished Creating Master File...')

    #load songs from master file
    print('Loading encoded songs...')
    songs = load_encoded_songs(ENCODED_PATH,'Master_File',0)
    print('Finished Loading encoded songs...')

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
    #in case we want to add the velocity
    #print(len(mapped_songs[4]))
    #print(len(mapped_songs[5]))
    #print(len(mapped_songs[6]))
    #print(len(mapped_songs[7]))
    print('-----')

    #this return a dictionary, with each key being a sequence, for both input and target
    inputs,targets = generate_training_sequences(mapped_songs,SEQUENCE_LENGTH, MAPPING_PATH)

    X = convert_dictionary_x(inputs)
    y = convert_dictionary_y(targets)

    #split the data into training,validating and testing
    len_data = X.shape[0]

    print(f'Total LEN:{len_data}')

    X_train = X[:-20,:,:]
    y_train = y[:-20,:,:]

    X_test = X[-20:-10,:,:]
    y_test = y[-20:-10,:,:]

    X_val = X[-10:,:,:]
    y_val = y[-10:,:,:]

    print(f'X_train:{X_train.shape}')
    print(f'y_train:{y_train.shape}')

    print(f'X_test:{X_test.shape}')
    print(f'y_test:{y_test.shape}')

    print(f'X_val:{X_val.shape}')
    print(f'y_val:{y_val.shape}')

    return X_train,y_train,X_test,y_test,X_val,y_val

if __name__ == "__main__":
    process_data()
