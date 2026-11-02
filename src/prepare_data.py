import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import (
    DATA_DIR, FEATURES_DIR, GENRES, SAMPLE_RATE, DURATION,
    N_MFCC, HOP_LENGTH, TARGET_FRAMES, RANDOM_STATE, TEST_SIZE
)


def extract_mfcc(path: Path) -> np.ndarray:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    expected = SAMPLE_RATE * DURATION
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    else:
        audio = audio[:expected]

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH
    )
    if mfcc.shape[1] < TARGET_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, TARGET_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :TARGET_FRAMES]
    return mfcc.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    X, y, paths = [], [], []
    for label, genre in enumerate(GENRES):
        genre_dir = args.data_dir / genre
        if not genre_dir.exists():
            raise FileNotFoundError(f"Missing genre directory: {genre_dir}")
        wavs = sorted([
            *genre_dir.glob("*.wav"),
            *genre_dir.glob("*.au"),
            *genre_dir.glob("*.AU")
        ])
        if not wavs:
            raise FileNotFoundError(f"No WAV files found in {genre_dir}")
        for wav in tqdm(wavs, desc=f"Processing {genre}"):
            try:
                X.append(extract_mfcc(wav))
                y.append(label)
                paths.append(str(wav))
            except Exception as exc:
                print(f"Skipping {wav}: {exc}")

    X = np.stack(X)
    y = np.asarray(y, dtype=np.int64)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    np.savez_compressed(
        FEATURES_DIR / "mfcc_dataset.npz",
        X=X, y=y, paths=np.asarray(paths),
        train_idx=train_idx, test_idx=test_idx,
        genres=np.asarray(GENRES)
    )
    metadata = {
        "samples": int(len(y)),
        "class_counts": {g: int((y == i).sum()) for i, g in enumerate(GENRES)},
        "mfcc_shape_per_sample": list(X.shape[1:]),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "genres": GENRES,
    }
    (FEATURES_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
