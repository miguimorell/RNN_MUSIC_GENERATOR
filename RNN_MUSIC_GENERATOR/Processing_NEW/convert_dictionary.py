import numpy as np

def convert_dictionary_x(dictionary):

    sequences = []
    for keys, values in dictionary.items():
        print(keys)
        print(values)
        for position in range(0,128):
            sequence = []
            for value in values:
                #print('-------')
                #print(value[position])
                sequence.append(value[position])
                #print('-------')
            sequences.append(sequence)

    print(sequences[0])

def convert_dictionary_x(dictionary):

    #print(dictionary)
    #sprint(len(dictionary))
    sequences = list(dictionary.values())
    train_data = np.array(sequences)
    # Rearrange the dimensions of the array
    train_data = np.transpose(train_data, (0, 2, 1, 3))
    print(train_data.shape)
    d0,d1,d2,d3 = train_data.shape
    train_data = np.reshape(train_data,(d0,d1,d2*d3))
    print(train_data.shape)
    return train_data

def convert_dictionary_y(dictionary):

    sequences = list(dictionary.values())
    train_data = np.array(sequences)
    # Rearrange the dimensions of the array
    train_data = np.transpose(train_data, (0, 2, 1))
    print(train_data.shape)
    return train_data