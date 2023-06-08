def convert_dictionary(variable):

    sequences = []
    for keys, values in variable.items():
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
