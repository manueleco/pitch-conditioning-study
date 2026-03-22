from pathlib import Path
import pandas as pd
import soundfile as sf

RAW_DIR = Path("data/raw/piano/fur_elise")
SYN_DIR = Path("data/synthetic/piano/fur_elise_aug")

OUT_POOL = Path("data/processed/manifest_pool.csv")

FINAL_AUDIO = Path("dataset/final/audio")
OUT_FINAL = Path("dataset/final/manifest.csv")

def build_manifest(audio_dirs, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in audio_dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.wav")):
            if not p.exists():
                continue
            info = sf.info(str(p))
            rows.append({
                "path": str(p),
                "filename": p.name,
                "samplerate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "duration_sec": info.frames / info.samplerate if info.samplerate else None,
                "source_dir": str(d),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"ok: {out_csv} ({len(df)} files)")

def main():
    build_manifest([RAW_DIR, SYN_DIR], OUT_POOL)
    build_manifest([FINAL_AUDIO], OUT_FINAL)

if __name__ == "__main__":
    main()