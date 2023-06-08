import keras
import json
import numpy as np

def generate_training_sequences(mapped_songs,sequence_length,mapping_path):
    """Create input and output data samples for training. Each sample is a sequence.

    :param sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars

    :return inputs (ndarray): Training inputs
    :return targets (ndarray): Training targets
    """

    inputs = {}
    targets = {}

    # generate the training sequences
    length = len(mapped_songs[0])
    num_sequences = length - sequence_length
    #print('Number of sequences:',num_sequences)
    for i in range(num_sequences):
       # print('i: ',i)
        for line in mapped_songs[-6:]:
            if i not in inputs: #initialize the key
                inputs[i] = []
            #print('X: ',line[i:i + sequence_length])
            #print('y: ',line[i + sequence_length])
            inputs[i].append(line[i:i + sequence_length])

        for line in mapped_songs[:2]:
            if i not in targets: #initialize the key
                targets[i] = []
            targets[i].append(line[i:i + sequence_length])

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
