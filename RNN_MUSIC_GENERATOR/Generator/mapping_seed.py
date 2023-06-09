import os
import json


MAPPING = {"r": 0,
    "59": 1,
    "61": 2,
    "70": 3,
    "38": 4,
    "34": 5,
    "39": 6,
    "46": 7,
    "53": 8,
    "36": 9,
    "63": 10,
    "29": 11,
    "44": 12,
    "/": 13,
    "30": 14,
    "65": 15,
    "69": 16,
    "47": 17,
    "62": 18,
    "48": 19,
    "43": 20,
    "68": 21,
    "54": 22,
    "50": 23,
    "31": 24,
    "52": 25,
    "37": 26,
    "51": 27,
    "40": 28,
    "_": 29,
    "41": 30,
    "42": 31,
    "32": 32,
    "49": 33,
    "66": 34,
    "45": 35,
    "67": 36,
    "57": 37,
    "64": 38,
    "35": 39,
    "33": 40,
    "58": 41,
    "60": 42,
    "56": 43
}

def map_seed(seed, MAPPING):

    # load mappings
    mapping = MAPPING

    int_songs2 = [[],[],[]]
    for index, instrument in enumerate(seed):
        for symbol in instrument:
            int_songs2[index].append(mapping[symbol])

    return int_songs2



if __name__ == "__main__":
    pass
