from pathlib import Path
import random
import numpy as np
import soundfile as sf
import librosa

IN_DIR = Path("data/raw/piano/fur_elise")
OUT_DIR = Path("data/synthetic/piano/fur_elise_aug")

# cuántas variantes por archivo
AUG_PER_FILE = 2

# rangos simples
TIME_STRETCH = (0.95, 1.05)     # tempo +/- 5%
PITCH_SHIFT = (-1.0, 1.0)       # semitonos
GAIN_DB = (-2.0, 2.0)           # dB
NOISE_STD = (0.0, 0.002)        # ruido muy bajo

SEED = 7

def db_to_amp(db):
    return 10 ** (db / 20)

def normalize_peak(y, peak=0.98):
    m = np.max(np.abs(y)) if len(y) else 0.0
    if m == 0:
        return y
    return y * (peak / m)

def add_noise(y, std):
    if std <= 0:
        return y
    n = np.random.normal(0.0, std, size=y.shape).astype(y.dtype)
    return y + n

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wavs = sorted(IN_DIR.glob("*.wav"))
    if not wavs:
        print("no wav files found:", IN_DIR)
        return

    total = 0
    for p in wavs:
        y, sr = librosa.load(p, sr=None, mono=True)

        for k in range(AUG_PER_FILE):
            y2 = y.copy()

            # time-stretch
            rate = random.uniform(*TIME_STRETCH)
            y2 = librosa.effects.time_stretch(y2, rate=rate)

            # pitch-shift
            n_steps = random.uniform(*PITCH_SHIFT)
            y2 = librosa.effects.pitch_shift(y2, sr=sr, n_steps=n_steps)

            # gain
            g = db_to_amp(random.uniform(*GAIN_DB))
            y2 = y2 * g

            # noise
            std = random.uniform(*NOISE_STD)
            y2 = add_noise(y2, std)

            # peak normalize (evita clipping feo)
            y2 = normalize_peak(y2, peak=0.98).astype(np.float32)

            out_name = f"{p.stem}__aug{k+1}_ts{rate:.3f}_ps{n_steps:.2f}.wav"
            out_path = OUT_DIR / out_name
            sf.write(out_path, y2, sr)
            total += 1

    print("ok:", total, "files ->", OUT_DIR)

if __name__ == "__main__":
    main()
