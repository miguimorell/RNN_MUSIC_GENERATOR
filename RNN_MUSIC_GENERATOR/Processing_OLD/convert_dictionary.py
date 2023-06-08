import numpy as np

def convert_dictionary(variable):

    kick = []
    vkick = []
    sn = []
    vsn = []
    hh = []
    vhh = []
    for keys, values in variable.items():
        for position in range(0,128):
            for ind, value in enumerate(values):
                #print(ind, '-------')
                #print(value[position])
                kick.append(value[position])
                vkick.append(value[position])
                sn.append(value[position])
                vsn.append(value[position])
                hh.append(value[position])
                vhh.append(value[position])
                #print('-------')

    #print(sequence[0])
    #print(sequences)
    output = np.array([kick,vkick,sn,vsn,hh,vhh]).astype(np.int16)
    print (output.shape)
    return output
