import os
import music21 as m21

def load_songs(dataset_path):
    """Loads all kern pieces in dataset using music21.

    :param dataset_path (str): Path to dataset
    :return songs (list of m21 streams): List containing all pieces
    """
    songs = []
    files_name = []

    # go through all the files in dataset and load them with music21
    for path, _, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == "mid":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
                files_name.append(file[-9:-4])
                #print(f'{file[-9:-4]} Song Loaded')
    return songs,files_name
