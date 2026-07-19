"""
STEP 3 & 4: Build a deep learning model (stacked LSTM) and train it
on the preprocessed dataset to learn music patterns.
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(42)
def build_model(seq_length, n_vocab):
    model = keras.Sequential([
        layers.Input(shape=(seq_length, 1)),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(256),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_vocab, activation="softmax"),
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def main():
    data = np.load("data/sequences.npz")
    X, y = data["X"], data["y"]
    with open("data/vocab.pkl", "rb") as f:
        vocab_info = pickle.load(f)
    n_vocab = vocab_info["n_vocab"]
    seq_length = vocab_info["sequence_length"]
    y_cat = to_categorical(y, num_classes=n_vocab)
    print(f"Training on {X.shape[0]} sequences | seq_len={seq_length} | vocab={n_vocab}")
    model = build_model(seq_length, n_vocab)
    model.summary()
    checkpoint = keras.callbacks.ModelCheckpoint(
        "models/best_model.keras",
        monitor="loss",
        save_best_only=True,
        verbose=1,
    )
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=15, restore_best_weights=True)
    history = model.fit(
        X, y_cat,
        epochs=40,          
        batch_size=64,
        callbacks=[checkpoint, early_stop],
        verbose=2,
    )
    model.save("models/final_model.keras")
    with open("models/history.pkl", "wb") as f:
        pickle.dump(history.history, f)
    print("\nSaved trained model to models/final_model.keras")

if __name__ == "__main__":
    main()
