"""
STEP 2: Preprocess MIDI data into note sequences suitable for training,
using music21.
"""
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

SEQUENCE_LENGTH = 32 
def get_notes_from_file(file_path):
    midi = converter.parse(file_path)
    tokens = []
    try:
        parts = instrument.partitionByInstrument(midi)
    except Exception:
        parts = None
    notes_to_parse = parts.parts[0].recurse() if parts else midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            tokens.append(".".join(str(n) for n in element.normalOrder))
    return tokens


def build_sequences(all_tokens, vocab_to_int):
    network_input = []
    network_output = []
    for i in range(len(all_tokens) - SEQUENCE_LENGTH):
        seq_in = all_tokens[i:i + SEQUENCE_LENGTH]
        seq_out = all_tokens[i + SEQUENCE_LENGTH]
        network_input.append([vocab_to_int[t] for t in seq_in])
        network_output.append(vocab_to_int[seq_out])
    return network_input, network_output


def main():
    midi_files = glob.glob("data/midi_songs/*.mid")
    print(f"Found {len(midi_files)} MIDI files.")
    all_tokens = []          
    per_song_tokens = []   
    for f in midi_files:
        print(f"Parsing {f} ...")
        tokens = get_notes_from_file(f)
        per_song_tokens.append(tokens)
        all_tokens.extend(tokens)
    print(f"Total tokens (notes/chords) parsed: {len(all_tokens)}")

    # Build vocabulary
    vocab = sorted(set(all_tokens))
    vocab_to_int = {token: i for i, token in enumerate(vocab)}
    int_to_vocab = {i: token for token, i in vocab_to_int.items()}
    n_vocab = len(vocab)
    print(f"Vocabulary size (unique notes/chords): {n_vocab}")

    # Build training sequences
    network_input, network_output = build_sequences(all_tokens, vocab_to_int)
    n_patterns = len(network_input)
    print(f"Total training sequences: {n_patterns}")

    # Reshape for LSTM
    X = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))
    X = X / float(n_vocab)
    y = np.array(network_output)

    # Save everything
    with open("data/notes.pkl", "wb") as f:
        pickle.dump({"all_tokens": all_tokens, "per_song_tokens": per_song_tokens}, f)
    with open("data/vocab.pkl", "wb") as f:
        pickle.dump({"vocab_to_int": vocab_to_int, "int_to_vocab": int_to_vocab,
                     "n_vocab": n_vocab, "sequence_length": SEQUENCE_LENGTH}, f)
    np.savez("data/sequences.npz", X=X, y=y)
    print("\nSaved: data/notes.pkl, data/vocab.pkl, data/sequences.npz")
    print(f"Input shape: {X.shape}  Output shape: {y.shape}")

if __name__ == "__main__":
    main()
