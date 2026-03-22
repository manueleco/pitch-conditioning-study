import random
import shutil
from pathlib import Path

SEED = 7
TRAIN = 0.7
VAL = 0.15
TEST = 0.15

FINAL_DIR = Path("dataset/final")
AUDIO_DIR = FINAL_DIR / "audio"
ANN_DIR = FINAL_DIR / "annotations"
PARAMS = FINAL_DIR / "parameters.json"

OUT_DIR = Path("dataset/splits")

def ensure_dirs(split_name: str):
    (OUT_DIR / split_name / "audio").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / split_name / "annotations").mkdir(parents=True, exist_ok=True)

def copy_pair(wav: Path, split_name: str):
    csv = ANN_DIR / (wav.stem + ".csv")
    if not csv.exists():
        return False

    out_wav = OUT_DIR / split_name / "audio" / wav.name
    out_csv = OUT_DIR / split_name / "annotations" / csv.name

    shutil.copy2(wav, out_wav)
    shutil.copy2(csv, out_csv)
    return True

def main():
    assert abs((TRAIN + VAL + TEST) - 1.0) < 1e-9

    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print("no wav files found:", AUDIO_DIR)
        return

    random.seed(SEED)
    random.shuffle(wavs)

    n = len(wavs)
    n_train = int(round(n * TRAIN))
    n_val = int(round(n * VAL))
    n_test = n - n_train - n_val

    splits = {
        "train": wavs[:n_train],
        "val": wavs[n_train:n_train + n_val],
        "test": wavs[n_train + n_val:],
    }

    for name in splits:
        ensure_dirs(name)

    counts = {"train": 0, "val": 0, "test": 0}
    for name, items in splits.items():
        for wav in items:
            if copy_pair(wav, name):
                counts[name] += 1

    # copy parameters.json to splits root (1 copy)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PARAMS.exists():
        shutil.copy2(PARAMS, OUT_DIR / "parameters.json")

    print("ok: splits")
    print("train:", counts["train"])
    print("val:", counts["val"])
    print("test:", counts["test"])
    print("out:", OUT_DIR)

if __name__ == "__main__":
    main()
