import os
import music21 as m21
from Data.load_midi import encode_song
import glob
import json
import keras
import numpy as np

'''
MIDI FILES GIVES ME THE FOLLOWING INFORMATION:
NOTA:C4 , DURATION: 0.25s, VELOCIDAD: 125
SILENCIO: , DURATION:
'''
DATASET_PATH = '/Users/Cris/code/miguimorell/RNN_MUSIC_GENERATOR/raw_data'
ENCODED_PATH = '/Users/Cris/code/miguimorell/RNN_MUSIC_GENERATOR/RNN_MUSIC_GENERATOR/Processing/Data/encoded'
ENCODED_PATH_INT = '/Users/Cris/code/miguimorell/RNN_MUSIC_GENERATOR/RNN_MUSIC_GENERATOR/Processing/Data/encoded'
MAPPING_PATH = '/Users/Cris/code/miguimorell/RNN_MUSIC_GENERATOR/RNN_MUSIC_GENERATOR/Processing/Data/mapping.json'
SEQUENCE_LENGTH = 128




def create_encoded_files(songs,files_name):
    for song,file_name in zip(songs,files_name):
        encoded_song,encoded_velocity=encode_song(song)
        # save songs to text file
        save_path = os.path.join(ENCODED_PATH, file_name)
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
            fp.write('\n')
            fp.write(encoded_velocity)

def create_master_file(sequence_length,encoded_path):
    # Define the prefix for each group of files
    prefixes = ['BS', 'CK', 'HH', 'SN']
    save_path = os.path.join(encoded_path, 'Master_File')
    delimeter = '/ ' * (sequence_length)

    with open(save_path, 'w') as output:
        for prefix in prefixes:
            files = glob.glob(os.path.join(encoded_path, prefix + '_*'))
            files.sort()

            line1 = []
            line2 = []
            # Merge the content from files into a single line
            for file_path in files:
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    line1.append(lines[0].rstrip('\n'))
                    line1.append(delimeter)
                    line2.append(lines[1].rstrip('\n'))
                    line2.append(delimeter)

            # Write the merged content to the output file
            output.write(' '.join(line1))
            output.write('\n')
            output.write(' '.join(line2))
            output.write('\n')

def create_mapping(songs, mapping_path):
    """Creates a json file that maps the symbols in the song dataset onto integers

    :param songs (str): String with all songs
    :param mapping_path (str): Path where to save mapping
    :return:
    """
    mappings = {}

    # identify the vocabulary
    songs_strings = " ".join(songs)
    songs = songs_strings.split()
    vocabulary = list(set(songs))

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # save voabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

def map_file(songs,mapping_path):

    # load mappings
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)

    save_path = os.path.join(ENCODED_PATH_INT, 'Master_File_int')
    # transform songs string to list
    for line in songs:
        int_songs = []

        line2 = line.split()

        # map songs to int
        for symbol in line2:
            int_songs.append(mappings[symbol])

        with open(save_path, "a") as fp:
            fp.write(str(int_songs))
            fp.write('\n')

def load_encoded_songs(encoded_path,file_name,option):
    if option == 0:
        load_path = os.path.join(encoded_path,file_name)
        with open(load_path, "r") as file:
            lines = file.readlines()
        return lines

    elif option == 1:
        load_path = os.path.join(encoded_path, file_name)
        mapped_songs = []
        with open(load_path, "r") as file:
            content = file.readlines()

            #print(content)
            # Remove square brackets and split the content based on comma delimiter
            for line in content:
                elements = line.replace("[", "").replace("]", "").split(",")
              #  print(elements)
            #Convert elements to integers
                elem = [int(element.strip()) for element in elements]
                mapped_songs.append(elem)
        return mapped_songs

def generate_training_sequences(mapped_songs,sequence_length):
    """Create input and output data samples for training. Each sample is a sequence.

    :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    inputs = {}
    targets = {}

    # generate the training sequences
    length = len(mapped_songs[0])
    print(len(mapped_songs[0]))
    num_sequences = length - sequence_length
    #print('Number of sequences:',num_sequences)
    for i in range(1):
        print('i: ',i)
        for line in mapped_songs:
            if i not in inputs: #initialize the key
                inputs[i] = []
                targets[i] = []
            #print('X: ',line[i:i + sequence_length])
            #print('y: ',line[i + sequence_length])
            inputs[i].append(line[i:i + sequence_length])
            targets[i].append(line[i + sequence_length])


    # one-hot encode the sequences
    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    vocabulary_size = len(mappings.keys())
    print(vocabulary_size)

    for key in inputs:
        sequences = inputs[key]
        encoded_sequences = []
        for sequence in sequences:
            encoded_sequence = []
            for element in sequence:
                encoded_element = keras.utils.to_categorical(element, num_classes=vocabulary_size,dtype='int32')
                encoded_sequence.append(encoded_element)
            encoded_sequences.append(encoded_sequence)

        inputs[key] = np.array(encoded_sequences)

    targets = np.array(targets)

    return inputs, targets

def main():

    songs,files_name = load_songs(DATASET_PATH)

    #generate encoded files
    create_encoded_files(songs,files_name)

    #create master file: 1 line BS, 2 line velocity,3 line KICK, 4 line velocity, etc. TOTAL 8 lines
    create_master_file(SEQUENCE_LENGTH,ENCODED_PATH)

    #load songs from master file
    songs = load_encoded_songs(ENCODED_PATH,'Master_File',0)

    #create file mapping with vocabulary to then convert characters to integers
    create_mapping(songs,MAPPING_PATH)

    #we map the master_file according to the vocabulary generated in the previous step
    map_file(songs,MAPPING_PATH)

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
    inputs,targets = generate_training_sequences(mapped_songs,SEQUENCE_LENGTH)

    #print(inputs.shape)
    #print(targets.shape)

if __name__ == "__main__":
    main()
