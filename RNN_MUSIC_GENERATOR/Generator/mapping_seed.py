import os
import json

def map_seed(seed,mapping_path):

    # load mappings
    with open(mapping_path, "r") as fp:
        mapping = json.load(fp)

    int_songs2 = [[],[],[]]
    for index, instrument in enumerate(seed):
        for symbol in instrument:
            int_songs2[index].append(mapping[symbol])

    return int_songs2



if __name__ == "__main__":
    pass
