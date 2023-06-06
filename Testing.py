
from music21 import converter, instrument, note, stream

def process_midi_file(midi_file_path):
    # Cargar el archivo MIDI
    midi = converter.parse(midi_file_path)

    # Obtener todas las notas y silencios del archivo MIDI
    notes_and_rests = []
    for element in midi.flat.getElementsByClass([note.Note, note.Rest]):
        if isinstance(element, note.Note):
            notes_and_rests.append(element)
        elif isinstance(element, note.Rest):
            notes_and_rests.append(element)

    # Extraer la información de las notas y silencios
    for element in notes_and_rests:
        if isinstance(element, note.Note):
            pitch = element.pitch
            duration = element.duration.quarterLength
            velocity = element.volume.velocity

            # Hacer algo con la información de la nota
            print(f"Nota: {pitch}, Duración: {duration}, Velocidad: {velocity}")
        elif isinstance(element, note.Rest):
            duration = element.duration.quarterLength

            # Hacer algo con la información del silencio
            print(f"Silencio, Duración: {duration}")

# Proporciona la ruta de tu archivo MIDI
midi_file_path="/home/miguimorell/code/miguimorell/RNN_MUSIC_GENERATOR/raw_data/Tanda_2/Miguel_SN_30.mid"

process_midi_file(midi_file_path)
