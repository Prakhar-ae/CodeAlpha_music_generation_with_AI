"""
STEP 5: Generate new music with the trained model and convert the
predicted sequence back into a MIDI file using music21.

"""

import pickle
import numpy as np
from tensorflow import keras
from music21 import stream, note, chord, instrument

SEQUENCE_LENGTH = 32
N_NOTES_TO_GENERATE = 200
TEMPERATURE = 3.0 

def sample_with_temperature(probs, temperature):
    probs = np.asarray(probs).astype("float64")
    probs = np.log(probs + 1e-9) / temperature
    exp_probs = np.exp(probs)
    probs = exp_probs / np.sum(exp_probs)
    return np.random.choice(len(probs), p=probs)

def generate_notes(model, seed_sequence, n_vocab, int_to_vocab, n_generate, temperature):
    pattern = list(seed_sequence)
    generated = []
    for _ in range(n_generate):
        input_seq = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        prediction = model.predict(input_seq, verbose=0)[0]
        next_index = sample_with_temperature(prediction, temperature)
        generated.append(int_to_vocab[next_index])
        pattern.append(next_index)
        pattern = pattern[1:]  
    return generated

def tokens_to_midi(tokens, out_path, step_duration=0.5):
    s = stream.Stream()
    s.append(instrument.Piano())
    offset = 0.0
    for token in tokens:
        if ("." in token) or token.isdigit():
            pitches = [int(p) for p in token.split(".")]
            notes = [note.Note(p + 60) for p in pitches] 
            c = chord.Chord(notes)
            c.offset = offset
            s.append(c)
        else:
            n = note.Note(token)
            n.offset = offset
            s.append(n)
        offset += step_duration

    s.write("midi", fp=out_path)


def main():
    with open("data/vocab.pkl", "rb") as f:
        vocab_info = pickle.load(f)
    int_to_vocab = vocab_info["int_to_vocab"]
    vocab_to_int = vocab_info["vocab_to_int"]
    n_vocab = vocab_info["n_vocab"]
    with open("data/notes.pkl", "rb") as f:
        notes_data = pickle.load(f)
    all_tokens = notes_data["all_tokens"]
    model = keras.models.load_model("models/final_model.keras")
    start = np.random.randint(0, len(all_tokens) - SEQUENCE_LENGTH - 1)
    seed_tokens = all_tokens[start:start + SEQUENCE_LENGTH]
    seed_sequence = [vocab_to_int[t] for t in seed_tokens]
    print(f"Seed sequence (from training data): {seed_tokens[:8]}...")
    print(f"Generating {N_NOTES_TO_GENERATE} new notes at temperature={TEMPERATURE} ...")
    generated_tokens = generate_notes(
        model, seed_sequence, n_vocab, int_to_vocab, N_NOTES_TO_GENERATE, TEMPERATURE
    )
    out_path = "output/generated_music.mid"
    tokens_to_midi(generated_tokens, out_path)
    print(f"\nSaved generated MIDI to {out_path}")
    print(f"First 20 generated tokens: {generated_tokens[:20]}")

if __name__ == "__main__":
    main()
