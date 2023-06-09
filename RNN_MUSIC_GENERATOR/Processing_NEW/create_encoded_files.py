from RNN_MUSIC_GENERATOR.Processing_NEW.load_midi import encode_song
import os

def create_encoded_files(songs,files_name,encoded_path):
    for song,file_name in zip(songs,files_name):
        encoded_song,encoded_velocity=encode_song(song)
        # save songs to text file
        save_path = os.path.join(encoded_path, file_name)
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
            if int(file_name[-2:]) == 26:
                print(len(encoded_song))
            fp.write('\n')
            fp.write(encoded_velocity)
