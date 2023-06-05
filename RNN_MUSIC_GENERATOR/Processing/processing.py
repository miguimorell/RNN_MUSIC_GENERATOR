import os
import music21 as m21
from Data.load_midi import encode_song
import glob

'''
MIDI FILES GIVES ME THE FOLLOWING INFORMATION:
NOTA:C4 , DURATION: 0.25s, VELOCIDAD: 125
SILENCIO: , DURATION:
'''
DATASET_PATH = '/Users/Cris/code/miguimorell/RNN_MUSIC_GENERATOR/raw_data'
ENCODED_PATH = '/Users/Cris/code/miguimorell/RNN_MUSIC_GENERATOR/RNN_MUSIC_GENERATOR/Processing/Data/encoded'
SEQUENCE_LENGTH = 128


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
    return songs,files_name

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

def main():

    songs,files_name = load_songs(DATASET_PATH)

    #generate encoded files
    create_encoded_files(songs,files_name)

    create_master_file(SEQUENCE_LENGTH,ENCODED_PATH)

if __name__ == "__main__":
    main()
