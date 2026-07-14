"""
make_libri3mix_test.py -- build the Libri3Mix (or Libri2Mix) *test* set for EVALUATION
from just LibriSpeech test-clean. No WHAM!, no SoX, no 332 GB.

Why this is safe: SI-SDRi evaluation only needs the TEST split, and a *clean* mixture
is exactly  sum_k( gain_k * resample(source_k) )  -- every gain and source path is in
the LibriMix metadata CSV. We reproduce the official LibriMix `mix_clean` math
(normalize by gain -> resample 16k->8k -> trim to the shortest source -> sum), and
skip the noise entirely.

Output layout (data_root = OUTDIR), matching what separate.py / evaluate.py expect:
    OUTDIR/mix_clean/<id>.wav
    OUTDIR/s1/<id>.wav  s2/<id>.wav  s3/<id>.wav
    OUTDIR/manifest_test.csv    # mixture_ID, mixture_path, source_k_path..., length

Then:
    python separate.py --model speechbrain/sepformer-libri3mix \
        --metadata OUTDIR/manifest_test.csv --data-root OUTDIR --outdir est/
    python evaluate.py --metadata OUTDIR/manifest_test.csv --data-root OUTDIR \
        --est-dir est/ --n-src 3

The SepFormer sanity gate (~19.8 dB SI-SDRi) doubles as validation that this
generator matches the data the model was trained on.

Usage:
    python make_libri3mix_test.py \
        --metadata .../libri3mix_test-clean.csv \
        --librispeech-root .../LibriSpeech \
        --outdir libri3mix_test --sr 8000 [--limit N]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf

LIBRISPEECH_RATE = 16000


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1-D audio. Prefers scipy.resample_poly (what official LibriMix uses);
    falls back to torchaudio. Imported lazily so the module loads even where scipy is
    broken (e.g. a botched local Anaconda) as long as no resampling is requested."""
    if orig_sr == target_sr:
        return audio
    try:
        from scipy.signal import resample_poly
        return resample_poly(audio, target_sr, orig_sr).astype(np.float32)
    except Exception:
        import torch
        import torchaudio
        out = torchaudio.functional.resample(torch.from_numpy(audio), orig_sr, target_sr)
        return out.numpy().astype(np.float32)


def n_src_from_columns(cols) -> int:
    return sum(1 for c in cols if c.startswith("source_") and c.endswith("_path"))


def build_clean_mixture(row, n_src: int, librispeech_root: str, sr: int):
    """Return (mixture, [source1..sourceN]) as 1-D float32 arrays at `sr`, min-length."""
    srcs = []
    for k in range(1, n_src + 1):
        path = os.path.join(librispeech_root, row[f"source_{k}_path"])
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        gain = float(row[f"source_{k}_gain"])
        audio = audio * gain                                   # loudness normalize
        audio = _resample(audio, file_sr, sr)                  # 16k -> 8k (no-op if equal)
        srcs.append(audio.astype(np.float32))

    # 'min' mode: trim every source to the shortest one, then sum
    target_len = min(len(s) for s in srcs)
    srcs = [s[:target_len] for s in srcs]
    mixture = np.sum(srcs, axis=0).astype(np.float32)
    return mixture, srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="libri{2,3}mix_test-clean.csv")
    ap.add_argument("--librispeech-root", required=True,
                    help="dir containing test-clean/ (the source .flac tree)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    md = pd.read_csv(args.metadata)
    if args.limit:
        md = md.head(args.limit)
    n_src = n_src_from_columns(md.columns)
    print(f"Generating {len(md)} test mixtures, n_src={n_src}, sr={args.sr}")

    for d in ["mix_clean"] + [f"s{k}" for k in range(1, n_src + 1)]:
        os.makedirs(os.path.join(args.outdir, d), exist_ok=True)

    records = []
    for i, (_, row) in enumerate(md.iterrows(), 1):
        mid = row["mixture_ID"]
        mixture, srcs = build_clean_mixture(row, n_src, args.librispeech_root, args.sr)

        mix_rel = os.path.join("mix_clean", f"{mid}.wav")
        # float32 (subtype FLOAT): mixtures can exceed +/-1.0; 16-bit PCM would clip
        # and quantize them, corrupting the model's input and the SI-SDR references.
        sf.write(os.path.join(args.outdir, mix_rel), mixture, args.sr, subtype="FLOAT")
        rec = {"mixture_ID": mid, "mixture_path": mix_rel, "length": len(mixture)}
        for k in range(1, n_src + 1):
            s_rel = os.path.join(f"s{k}", f"{mid}.wav")
            sf.write(os.path.join(args.outdir, s_rel), srcs[k - 1], args.sr, subtype="FLOAT")
            rec[f"source_{k}_path"] = s_rel
        records.append(rec)
        if i % 200 == 0:
            print(f"  ...{i}/{len(md)}")

    manifest = os.path.join(args.outdir, "manifest_test.csv")
    pd.DataFrame.from_records(records).to_csv(manifest, index=False)
    print(f"Done. Wrote {len(records)} mixtures + {manifest}")


if __name__ == "__main__":
    main()
