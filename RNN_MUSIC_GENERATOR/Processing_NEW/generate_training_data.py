import keras
import json
import numpy as np
import os

MAPPING_PATH = os.environ.get("MAPPING_PATH")


def generate_training_sequences(mapped_songs,sequence_length,mapping_path):
    """Create input and output data samples for training. Each sample is a sequence.

    :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    inputs = {}
    targets = {}

    #because the mapping changes every time we code is run, the number of classes may change
    #if more were added when song were added. So we need the lenght of the dictionary
    #for the last layer

    # load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    key_value = mappings['/']
    print(f'Key_Value: {key_value}')

    # generate the training sequences
    length = len(mapped_songs[0])
    print('total length')
    print(length)
    num_sequences = length - sequence_length
    for i in range(num_sequences):
        #in case we want to add the velocity
        #for line in mapped_songs[-6:]:
        for line in mapped_songs[-3:]:
            line_to_read = line[i:i + sequence_length]
            if key_value in line_to_read:
                continue
            if i not in inputs: #initialize the key
                inputs[i] = []
            inputs[i].append(line_to_read)

        #print(f'Key {i}, sequences: {cant_muestras_por_cancion}')
        for line in mapped_songs[:1]:
        #in case we want to add the velocity
        #for line in mapped_songs[:2]:
            line_to_read = line[i:i + sequence_length]
            if key_value in line_to_read:
                continue
            if i not in targets: #initialize the key
                targets[i] = []

            targets[i].append(line_to_read)

    # one-hot encode the sequences
    # load mappings
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)

    vocabulary_size = len(mappings.keys())

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

    for key in targets:
        targets[key] = np.array(targets[key])

    return inputs, targets
