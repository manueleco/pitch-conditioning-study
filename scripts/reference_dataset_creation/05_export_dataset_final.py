import re
import random
import shutil
from pathlib import Path

import soundfile as sf

FPS = 75
TARGET_MINUTES = 12.0  # cámbialo a 10-15 si quieres
SEED = 7

RAW_DIR = Path("data/raw/piano/fur_elise")
SYN_DIR = Path("data/synthetic/piano/fur_elise_aug")

OUT_DIR = Path("dataset/final")
OUT_AUDIO = OUT_DIR / "audio"
OUT_ANN = OUT_DIR / "annotations"

PARAMS_SRC = Path("dataset/parameters.json")
PARAMS_DST = OUT_DIR / "parameters.json"

RE_TS = re.compile(r"_ts([0-9]+\.[0-9]+)")
RE_PS = re.compile(r"_ps(-?[0-9]+\.[0-9]+)")

def parse_params(stem: str):
    ts_m = RE_TS.search(stem)
    ps_m = RE_PS.search(stem)
    ts = float(ts_m.group(1)) if ts_m else 1.0
    ps = float(ps_m.group(1)) if ps_m else 0.0
    return ts, ps

def duration_sec(wav: Path) -> float:
    info = sf.info(str(wav))
    return info.frames / info.samplerate if info.samplerate else 0.0

def get_pairs(folder: Path):
    pairs = []
    for wav in sorted(folder.glob("*.wav")):
        csv = wav.with_suffix(".csv")
        if csv.exists():
            ts, ps = parse_params(wav.stem)
            pairs.append({
                "wav": wav,
                "csv": csv,
                "ts": ts,
                "ps": ps,
                "dur": duration_sec(wav),
                "src": folder.name,
            })
    return pairs

def bin_label(ts: float, ps: float):
    # 3 bins para ts y 3 bins para ps -> 9 grupos
    if ts < 0.983:
        tsb = "ts_low"
    elif ts > 1.017:
        tsb = "ts_high"
    else:
        tsb = "ts_mid"

    if ps < -0.33:
        psb = "ps_neg"
    elif ps > 0.33:
        psb = "ps_pos"
    else:
        psb = "ps_mid"

    return f"{tsb}__{psb}"

def select_balanced(items, target_sec):
    # agrupa por bins
    bins = {}
    for it in items:
        b = bin_label(it["ts"], it["ps"])
        bins.setdefault(b, []).append(it)

    # shuffle por bin
    random.seed(SEED)
    for b in bins:
        random.shuffle(bins[b])

    chosen = []
    total = 0.0

    # round-robin por bins hasta llegar
    bin_keys = sorted(bins.keys())
    idx = 0
    while total < target_sec:
        if not bin_keys:
            break

        b = bin_keys[idx % len(bin_keys)]
        idx += 1

        if not bins[b]:
            # si se vacía un bin, lo quitamos
            bin_keys = [k for k in bin_keys if bins[k]]
            idx = 0
            continue

        it = bins[b].pop()
        if it["dur"] <= 0:
            continue

        chosen.append(it)
        total += it["dur"]

        # stop suave si ya pasamos
        if total >= target_sec:
            break

    return chosen, total

def main():
    OUT_AUDIO.mkdir(parents=True, exist_ok=True)
    OUT_ANN.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # copia parameters.json
    if PARAMS_SRC.exists():
        shutil.copy2(PARAMS_SRC, PARAMS_DST)

    raw = get_pairs(RAW_DIR)
    syn = get_pairs(SYN_DIR)

    # mix: 50/50 aprox en tiempo
    target_sec = TARGET_MINUTES * 60.0
    target_raw = target_sec * 0.5
    target_syn = target_sec * 0.5

    chosen_raw, total_raw = select_balanced(raw, target_raw)
    chosen_syn, total_syn = select_balanced(syn, target_syn)

    chosen = chosen_raw + chosen_syn
    random.shuffle(chosen)

    # export
    exported = 0
    for it in chosen:
        wav = it["wav"]
        csv = it["csv"]

        out_wav = OUT_AUDIO / wav.name
        out_csv = OUT_ANN / (wav.stem + ".csv")

        shutil.copy2(wav, out_wav)
        shutil.copy2(csv, out_csv)
        exported += 1

    total = total_raw + total_syn

    print("ok: exported files:", exported)
    print("ok: total minutes:", round(total / 60.0, 2))
    print("ok: raw minutes:", round(total_raw / 60.0, 2))
    print("ok: syn minutes:", round(total_syn / 60.0, 2))
    print("out:", OUT_DIR)

if __name__ == "__main__":
    main()
