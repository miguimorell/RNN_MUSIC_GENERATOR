import os

def load_encoded_songs(encoded_path,file_name,option):
    if option == 0:
        load_path = os.path.join(encoded_path,file_name)
        with open(load_path, "r") as file:
            lines = file.readlines()
        return lines

    elif option == 1:
        load_path = os.path.join(encoded_path, file_name)
        mapped_songs = []
        with open(load_path, "r") as file:
            content = file.readlines()

            #print(content)
            # Remove square brackets and split the content based on comma delimiter
            for line in content:
                elements = line.replace("[", "").replace("]", "").split(",")
              #  print(elements)
            #Convert elements to integers
                elem = [int(element.strip()) for element in elements]

                #print(f'Element: {elem}')
                #print(f'len: {len(elem)}')
                mapped_songs.append(elem)
        return mapped_songs
