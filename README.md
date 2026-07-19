# Task — Music Generation with AI

An end-to-end LSTM pipeline that learns from MIDI music and generates new
note sequences, saved back out as playable MIDI.

## Pipeline (run in order)

| Script | Task step | What it does |
|---|---|---|
| `00_generate_sample_data.py` | Collect MIDI data | Generates a small demo MIDI corpus in `data/midi_songs/`. **Replace this folder with your own classical/jazz MIDI files** for real training. |
| `01_preprocess.py` | Preprocess | Uses `music21` to parse MIDI → note/chord token sequences → builds vocabulary → builds fixed-length (32-step) training windows → saves `data/notes.pkl`, `data/vocab.pkl`, `data/sequences.npz` |
| `02_train_model.py` | Build & train model | Stacked LSTM (3x LSTM(256) + Dense layers) trained on next-note prediction (categorical cross-entropy). Saves `models/final_model.keras` and `models/best_model.keras` (best checkpoint). |
| `03_generate_music.py` | Generate & convert to MIDI | Seeds from a real training window, autoregressively samples new notes/chords (temperature-controlled), converts back to MIDI via `music21`, saves `output/generated_music.mid` |

## Quick start

```bash
pip install music21 tensorflow
python3 00_generate_sample_data.py  
python3 01_preprocess.py
python3 02_train_model.py
python3 03_generate_music.py
```

Output: `output/generated_music.mid` — open in any DAW (GarageBand, MuseScore,
FL Studio, Ableton) or convert to audio with a soundfont/synth (e.g. `fluidsynth`,
or import into MuseScore and export to WAV/MP3).

## Notes on this demo run

Since no MIDI dataset was uploaded, `00_generate_sample_data.py` synthesizes
8 short demo songs from simple major/minor arpeggios so the full pipeline
(preprocess → train → generate → MIDI) could be demonstrated end-to-end.
The trained demo model has learned the sample corpus reasonably well (training
loss dropped from ~2.7 to ~2.1 over 40 epochs on this tiny dataset), but for
genuinely musical results you'll want a real dataset of at least a few dozen
full-length MIDI songs.
