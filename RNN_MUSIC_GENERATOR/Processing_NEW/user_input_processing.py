def translate_input(user_input):


    #Create simbols:  In a future cambiar por una llamada a un mapping
    send_kick_data = []
    send_snare_data = []
    send_charles_data = []
    for lista, sound in enumerate(user_input):
        for index, data in enumerate(sound):
            if lista == 0:
                if data == True:
                    send_kick_data.append("36")
                else:
                    send_kick_data.append("r")
            elif lista == 1:
                if data == True:
                    send_snare_data.append("38")
                else:
                    send_snare_data.append("r")
            elif lista == 2:
                if data == True:
                    send_charles_data.append("42")
                else:
                    send_charles_data.append("r")

    user_input = [send_kick_data, send_snare_data, send_charles_data]


    #Create underscores"_"
    modified_input = []
    for lista in user_input:
        modified_lista = []
        prev_element = None
        count_r = 0

        for element in lista:
            if element == "r":
                count_r += 1
                if count_r == 1:
                    modified_lista.append(element)
                else:
                    modified_lista.append("_")
            else:
                modified_lista.append(element)
                count_r = 0

        modified_input.append(modified_lista)

    return modified_input

if __name__ == "__main__":
    pass
