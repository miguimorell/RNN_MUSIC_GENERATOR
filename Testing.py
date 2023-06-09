
from music21 import converter, instrument, note, stream


#Funcion que carga el archivo midi, lee las notas y silencios y devuelve la duración total.
def process_midi_file(midi_file_path):
    # Cargar el archivo MIDI
    midi = converter.parse(midi_file_path)
    print(f"Archivo:{midi_file_path}")
    # Obtener todas las notas y silencios del archivo MIDI
    notes_and_rests = []
    for element in midi.flat.getElementsByClass([note.Note, note.Rest]):
        if isinstance(element, note.Note):
            notes_and_rests.append(element)
        elif isinstance(element, note.Rest):
            notes_and_rests.append(element)
    suma_duracion=0
    # Extraer la información de las notas y silencios
    for element in notes_and_rests:
        if isinstance(element, note.Note):
            pitch = element.pitch.midi
            duration = element.duration.quarterLength
            suma_duracion=suma_duracion+duration
            velocity = element.volume.velocity

            # Hacer algo con la información de la nota
            print(f"Nota: {pitch}, Duración: {duration}, Velocidad: {velocity}")
        elif isinstance(element, note.Rest):
            duration = element.duration.quarterLength
            suma_duracion=suma_duracion+duration

            # Hacer algo con la información del silencio
            print(f"Silencio, Duración: {duration}")
            print(f"El numero de elementos es : {len(notes_and_rests)}")
    return suma_duracion

#Path general
midi_file_path="/Users/Cris/code/miguimorell/RNN_MUSIC_GENERATOR/raw_data/Miguel_"

#Inicializacion variables
duration_kick=[]
duration_snare=[]
duration_charles=[]
duration_bass=[]

warning_kick=[]
warning_snare=[]
warning_charles=[]
warning_bass=[]



#Juntar todos los Kicks, snares, charles y bass y mandarlos a la funcion que lee
for file in range(25,31):

    midi_file_path_kick="".join([midi_file_path, f"KICK_{file}.mid"])

    midi_file_path_snare="".join([midi_file_path, f"SN_{file}.mid"])

    midi_file_path_charles="".join([midi_file_path, f"CH_{file}.mid"])

    midi_file_path_bass="".join([midi_file_path, f"BS_{file}.mid"])

    suma_duration_kick=0
    suma_duration_snare=0
    suma_duration_charles=0
    suma_duration_bass=0



    print("DATOS KICK:")
    print("----------------------------------------------------------------")
    suma_duration_kick=process_midi_file(midi_file_path_kick)
    print(f"Suma_duration kick:{suma_duration_kick}")
    duration_kick.append(suma_duration_kick)


    print("DATOS SNARE:")
    print("----------------------------------------------------------------")
    suma_duration_snare=process_midi_file(midi_file_path_snare)
    print(f"Suma_duration snare:{suma_duration_snare}")
    duration_snare.append(suma_duration_snare)

    print("DATOS CHARLES:")
    print("----------------------------------------------------------------")
    suma_duration_charles=process_midi_file(midi_file_path_charles)
    print(f"Suma_duration charles:{suma_duration_charles}")
    duration_charles.append(suma_duration_charles)


    print("DATOS BASS:")
    print("----------------------------------------------------------------")
    suma_duration_bass=process_midi_file(midi_file_path_bass)
    print(f"Suma_duration bass:{suma_duration_bass}")

    duration_bass.append(suma_duration_bass)


for index, element in enumerate(duration_kick):
    if element!= 32:
        warning_kick.append(index+1)

for index, element in enumerate(duration_snare):
    if element!= 32:
        warning_snare.append(index+1)

for index, element in enumerate(duration_charles):
    if element!= 32:
        warning_charles.append(index+1)


for index, element in enumerate(duration_bass):
    if element!= 32:
        warning_bass.append(index+1)


print("Warning Kick:")
print(warning_kick)

print("Warning Snare:")
print(warning_snare)

print("Warning Charles:")
print(warning_charles)

print("Warning Bass:")
print(warning_bass)
