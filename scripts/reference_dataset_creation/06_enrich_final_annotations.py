import pandas as pd
import numpy as np
import librosa
from pathlib import Path

FPS = 75

FINAL = Path("dataset/final")
AUDIO_DIR = FINAL / "audio"
ANN_DIR = FINAL / "annotations"

HEADER = "time_stretch,pitch_shift,loudness_rms,spectral_centroid,source\n"

def compute_features(y, sr):
    hop_length = max(1, int(round(sr / FPS)))
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length, center=True)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=hop_length, center=True)[0]
    n = int(min(len(rms), len(centroid)))
    return rms[:n], centroid[:n], hop_length

def main():
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print("no wavs:", AUDIO_DIR)
        return

    updated = 0
    for wav in wavs:
        csv_in = ANN_DIR / (wav.stem + ".csv")
        if not csv_in.exists():
            continue

        df0 = pd.read_csv(csv_in)
        ts = float(df0["time_stretch"].iloc[0])
        ps = float(df0["pitch_shift"].iloc[0])
        source_id = int(df0["source"].iloc[0])

        y, sr = librosa.load(wav, sr=None, mono=True)
        rms, centroid, _ = compute_features(y, sr)

        out = ANN_DIR / (wav.stem + ".csv")
        with out.open("w", encoding="utf-8") as f:
            f.write(HEADER)
            for i in range(len(rms)):
                f.write(f"{ts},{ps},{float(rms[i])},{float(centroid[i])},{source_id}\n")

        updated += 1

    print("ok: enriched", updated, "files")

if __name__ == "__main__":
    main()
