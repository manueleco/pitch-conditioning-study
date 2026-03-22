import re
from pathlib import Path
import soundfile as sf

FPS = 75

RAW_DIR = Path("data/raw/piano/fur_elise")
SYN_DIR = Path("data/synthetic/piano/fur_elise_aug")

RE_TS = re.compile(r"_ts([0-9]+\.[0-9]+)")
RE_PS = re.compile(r"_ps(-?[0-9]+\.[0-9]+)")

HEADER = "time_stretch,pitch_shift,source\n"

def parse_params(stem: str):
    ts_m = RE_TS.search(stem)
    ps_m = RE_PS.search(stem)
    ts = float(ts_m.group(1)) if ts_m else 1.0
    ps = float(ps_m.group(1)) if ps_m else 0.0
    return ts, ps

def write_csv(csv_path: Path, n_frames: int, ts: float, ps: float, source_id: int):
    line = f"{ts},{ps},{source_id}\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(HEADER)
        for _ in range(n_frames):
            f.write(line)

def process_dir(d: Path, source_id: int):
    if not d.exists():
        return 0
    count = 0
    for wav in sorted(d.glob("*.wav")):
        if not wav.exists():
            continue
        info = sf.info(str(wav))
        dur = info.frames / info.samplerate if info.samplerate else 0.0
        n_frames = max(1, int(round(dur * FPS)))

        ts, ps = parse_params(wav.stem)
        csv_path = wav.with_suffix(".csv")
        write_csv(csv_path, n_frames, ts, ps, source_id)
        count += 1
    return count

def main():
    n_raw = process_dir(RAW_DIR, 0)
    n_syn = process_dir(SYN_DIR, 1)
    print("ok: csv created for", n_raw + n_syn, "files")

if __name__ == "__main__":
    main()
