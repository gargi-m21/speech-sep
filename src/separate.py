"""
separate.py -- run a PRETRAINED separation model on mixtures and write one wav per
speaker (Person A, the "floor").

This is the guaranteed-working pipeline: input mixture -> N clean source wavs.
It writes outputs as  <outdir>/<mixture_id>_sK.wav  (K = 1..N), which is exactly the
naming convention evaluate.py expects -- so separation and scoring click together.

Backends
--------
  sepformer   : speechbrain SepFormer, pretrained, 8 kHz, FIXED speaker count
                  speechbrain/sepformer-libri2mix   (2 speakers)
                  speechbrain/sepformer-libri3mix   (3 speakers)
  (mdd)       : MultiDecoderDPRNN, UNKNOWN speaker count (2-5) -- added Day 2.

Runs on GPU (Kaggle) or CPU. On CPU it works but is slow -- fine for a smoke test
on a handful of files, use Kaggle for the full test set.

Usage
-----
  # one file
  python separate.py --model speechbrain/sepformer-libri3mix \
                     --input mix.wav --outdir out/

  # a whole LibriMix test set (drives evaluate.py afterwards)
  python separate.py --model speechbrain/sepformer-libri2mix \
                     --metadata <meta>.csv --data-root <root> --outdir out/
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch

# Monkey-patch torch.load to set weights_only=False by default.
# Required in PyTorch 2.6+ to load older Asteroid/PyTorch-Lightning checkpoints.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

TARGET_SR = 8000  # all our pretrained separators operate at 8 kHz


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def load_mono_8k(path: str) -> torch.Tensor:
    """Load a wav, downmix to mono, resample to 8 kHz. Returns 1-D float32 tensor."""
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav)
    if sr != TARGET_SR:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav


def write_sources(mixture_id: str, sources: torch.Tensor, outdir: str) -> list[str]:
    """
    sources: [n_src, time] tensor. Writes <outdir>/<id>_sK.wav for K=1..n_src.
    Returns the list of written paths.
    """
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for k in range(sources.shape[0]):
        p = os.path.join(outdir, f"{mixture_id}_s{k + 1}.wav")
        # float32: model outputs are NOT normalized and can exceed +/-1.0. Default
        # 16-bit PCM would clip those peaks -> non-linear distortion that tanks SI-SDR.
        sf.write(p, sources[k].detach().cpu().numpy(), TARGET_SR, subtype="FLOAT")
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Backend: SepFormer (speechbrain)
# --------------------------------------------------------------------------- #
class SepFormerBackend:
    """Wraps a pretrained speechbrain SepFormer separator."""

    def __init__(self, source: str, device: str):
        from speechbrain.inference.separation import SepformerSeparation
        self.model = SepformerSeparation.from_hparams(
            source=source,
            savedir=os.path.join("pretrained_models", source.split("/")[-1]),
            run_opts={"device": device},
        )
        self.device = device

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """mixture: 1-D 8 kHz tensor -> [n_src, time] tensor of estimated sources."""
        mix = mixture.to(self.device).unsqueeze(0)          # [1, time]
        est = self.model.separate_batch(mix)                # [1, time, n_src]
        return est.squeeze(0).transpose(0, 1).contiguous()  # [n_src, time]


class MultiDecoderDPRNNBackend:
    """Wraps a pretrained asteroid MultiDecoderDPRNN separator."""

    def __init__(self, source: str, device: str):
        from asteroid.models import BaseModel
        self.model = BaseModel.from_pretrained(source).to(device)
        self.device = device

    @torch.no_grad()
    def separate(self, mixture: torch.Tensor) -> torch.Tensor:
        """mixture: 1-D 8 kHz tensor -> [n_src, time] tensor of estimated sources."""
        mix = mixture.to(self.device)
        est = self.model.separate(mix)
        if isinstance(est, np.ndarray):
            est = torch.from_numpy(est)
        return est


def build_backend(model: str, device: str):
    if "sepformer" in model:
        return SepFormerBackend(model, device)
    elif "MultiDecoderDPRNN" in model:
        return MultiDecoderDPRNNBackend(model, device)
    raise ValueError(f"Unknown/not-yet-implemented model '{model}'")


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def separate_one(backend, input_path: str, outdir: str, mixture_id: str | None = None):
    mid = mixture_id or os.path.splitext(os.path.basename(input_path))[0]
    mix = load_mono_8k(input_path)
    sources = backend.separate(mix)
    paths = write_sources(mid, sources, outdir)
    print(f"[{mid}] wrote {len(paths)} sources -> {outdir}")
    return paths


def separate_librimix(backend, metadata_csv: str, data_root: str, outdir: str,
                      limit: int | None = None):
    md = pd.read_csv(metadata_csv)
    if limit:
        md = md.head(limit)
    for i, (_, m) in enumerate(md.iterrows(), 1):
        mid = m["mixture_ID"]
        mix_path = os.path.join(data_root, m["mixture_path"])
        separate_one(backend, mix_path, outdir, mixture_id=mid)
        if i % 25 == 0:
            print(f"  ...{i}/{len(md)}")
    print(f"Done: {len(md)} mixtures separated into {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Run a pretrained separator")
    ap.add_argument("--model", required=True,
                    help="e.g. speechbrain/sepformer-libri3mix")
    ap.add_argument("--input", help="single mixture wav")
    ap.add_argument("--metadata", help="LibriMix metadata csv (batch mode)")
    ap.add_argument("--data-root", help="dir the metadata paths are relative to")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only first N mixtures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"Loading {args.model} on {args.device} ...")
    backend = build_backend(args.model, args.device)

    if args.input:
        separate_one(backend, args.input, args.outdir)
    elif args.metadata:
        assert args.data_root, "--data-root required in batch mode"
        separate_librimix(backend, args.metadata, args.data_root, args.outdir, args.limit)
    else:
        raise SystemExit("Provide --input or --metadata")


if __name__ == "__main__":
    main()
