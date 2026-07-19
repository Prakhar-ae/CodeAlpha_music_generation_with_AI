"""
STEP 1: generating MIDI music data.

"""
import os
import random
from music21 import stream, note, chord, duration, tempo

random.seed(42)

OUT_DIR = "data/midi_songs"
os.makedirs(OUT_DIR, exist_ok=True)

PROGRESSIONS = [
    ["C4", "E4", "G4", "C5"],       
    ["A3", "C4", "E4", "A4"],         
    ["F3", "A3", "C4", "F4"],          
    ["G3", "B3", "D4", "G4"],          
    ["D4", "F4", "A4", "D5"],          
]

CHORDS = [
    ["C3", "E3", "G3"],
    ["A2", "C3", "E3"],
    ["F2", "A2", "C3"],
    ["G2", "B2", "D3"],
]

DURATIONS = [0.25, 0.5, 0.5, 1.0] 

def make_song(n_measures=16, swing=False):
    s = stream.Stream()
    s.append(tempo.MetronomeMark(number=100 if not swing else 120))
    for _ in range(n_measures):
        prog = random.choice(PROGRESSIONS)
        # melodic line
        for pitch_name in prog:
            n = note.Note(pitch_name)
            n.duration = duration.Duration(random.choice(DURATIONS))
            s.append(n)
        # occasional backing chord
        if random.random() < 0.5:
            c = chord.Chord(random.choice(CHORDS))
            c.duration = duration.Duration(1.0)
            s.append(c)
    return s


def main():
    n_files = 8
    for i in range(n_files):
        song = make_song(n_measures=20, swing=(i % 2 == 0))
        path = os.path.join(OUT_DIR, f"sample_song_{i+1}.mid")
        song.write("midi", fp=path)
        print(f"Wrote {path}")
    print(f"\nGenerated {n_files} sample MIDI files in '{OUT_DIR}/'")
    
if __name__ == "__main__":
    main()


