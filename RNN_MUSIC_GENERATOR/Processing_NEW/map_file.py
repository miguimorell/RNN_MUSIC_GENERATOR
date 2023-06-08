import os

def map_file(songs,mapping_path,encoded_path_int):

    # load mappings
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)

    save_path = os.path.join(encoded_path_int, 'Master_File_int')
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
