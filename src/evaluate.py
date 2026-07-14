"""
evaluate.py -- SI-SDRi evaluation harness for speech separation (Person A).

This is the team's "ruler": it scores ANY separation model's outputs against the
ground-truth source waveforms, so MTC-Net, SepTDA and the pretrained baselines are
all measured the exact same way.

------------------------------------------------------------------------------
Two ideas you must be able to explain to a reviewer:

1) SI-SDR  (Scale-Invariant Signal-to-Distortion Ratio, in dB)
   "How much of my estimate is the true voice vs. distortion, ignoring loudness?"
   We project the estimate onto the reference to find the best-scaled copy of the
   target (that projection is what makes it *scale-invariant* -- a 2x louder
   estimate scores the same). Then:  SI-SDR = 10*log10( target_energy / error_energy ).
   Higher dB = cleaner separation.

2) PIT  (Permutation-Invariant matching)
   The model outputs voices in an arbitrary order. If it perfectly recovers both
   speakers but labels them (2,1) instead of (1,2), we must NOT punish it. So we
   try every ordering of estimates against references and keep the best-scoring one.
   (N! permutations; fine for the N<=5 we care about.)

SI-SDRi ("i" = improvement) = SI-SDR(estimate) - SI-SDR(mixture).
   i.e. how much better than *doing nothing* (using the raw mixture as the guess).
   This is the number we report per speaker-count level (2/3/4/5).
------------------------------------------------------------------------------

Dependencies: torch, numpy, soundfile, pandas  (all preinstalled on Kaggle).
Runs on CPU -- evaluation needs no GPU.
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Audio I/O
# --------------------------------------------------------------------------- #
def load_wav(path: str) -> torch.Tensor:
    """Load a mono waveform as a 1-D float32 torch tensor."""
    wav, _sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:            # stereo -> take first channel
        wav = wav[:, 0]
    return torch.from_numpy(wav)


def _match_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim two signals to the same length (models can be off by a few samples)."""
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #
def si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    """
    Scale-Invariant SDR (dB) between one estimate and one reference (1-D tensors).

    Steps (this is the whole formula, deliberately unpacked so it's explainable):
      1. zero-mean both signals
      2. s_target = <est, ref> / <ref, ref> * ref     (best-scaled copy of the target)
      3. e_noise  = est - s_target                     (everything that isn't the target)
      4. SI-SDR   = 10*log10( ||s_target||^2 / ||e_noise||^2 )
    """
    est, ref = _match_length(estimate, reference)
    est = est - est.mean()
    ref = ref - ref.mean()

    # scalar projection coefficient alpha = <est,ref>/<ref,ref>
    alpha = torch.dot(est, ref) / (torch.dot(ref, ref) + EPS)
    s_target = alpha * ref
    e_noise = est - s_target

    ratio = torch.sum(s_target ** 2) / (torch.sum(e_noise ** 2) + EPS)
    return float(10.0 * torch.log10(ratio + EPS))


def best_permutation(estimates: list[torch.Tensor],
                     references: list[torch.Tensor]) -> tuple[float, tuple[int, ...], list[float]]:
    """
    PIT: try every ordering of `estimates` against `references`, return the ordering
    with the highest MEAN SI-SDR.

    Returns (best_mean_si_sdr, best_perm, per_reference_si_sdr_list)
    where per_reference_si_sdr_list[i] is the SI-SDR for reference i under the winning
    permutation.
    """
    n = len(references)
    assert len(estimates) == n, "PIT needs equal #estimates and #references"

    best_mean = -np.inf
    best_perm = tuple(range(n))
    best_scores: list[float] = []

    for perm in itertools.permutations(range(n)):
        scores = [si_sdr(estimates[perm[i]], references[i]) for i in range(n)]
        mean = float(np.mean(scores))
        if mean > best_mean:
            best_mean, best_perm, best_scores = mean, perm, scores
    return best_mean, best_perm, best_scores


def si_sdri_utterance(estimates: list[torch.Tensor],
                      references: list[torch.Tensor],
                      mixture: torch.Tensor) -> dict:
    """
    SI-SDRi for one mixture.

    - Match estimates to references with PIT.
    - Baseline = SI-SDR of the raw MIXTURE against each reference (doing nothing).
    - Improvement per reference = matched SI-SDR - mixture SI-SDR.
    - Utterance SI-SDRi = mean improvement over the references.
    """
    _, perm, matched_scores = best_permutation(estimates, references)
    baseline = [si_sdr(mixture, ref) for ref in references]
    improvements = [matched_scores[i] - baseline[i] for i in range(len(references))]
    return {
        "n_src": len(references),
        "si_sdr": float(np.mean(matched_scores)),        # absolute quality
        "si_sdr_mixture": float(np.mean(baseline)),      # baseline (input)
        "si_sdri": float(np.mean(improvements)),         # the headline number
        "perm": perm,
    }


# --------------------------------------------------------------------------- #
# Batch evaluation over a manifest
# --------------------------------------------------------------------------- #
def evaluate_manifest(rows: list[dict]) -> pd.DataFrame:
    """
    rows: list of dicts, each with
        mixture_id   : str
        mixture_path : str
        ref_paths    : list[str]   (ground-truth s1..sN)
        est_paths    : list[str]   (model outputs, any order)
    Returns a per-utterance DataFrame with si_sdr / si_sdr_mixture / si_sdri / n_src.
    """
    records = []
    for r in rows:
        refs = [load_wav(p) for p in r["ref_paths"]]
        ests = [load_wav(p) for p in r["est_paths"]]
        mix = load_wav(r["mixture_path"])
        res = si_sdri_utterance(ests, refs, mix)
        res["mixture_id"] = r["mixture_id"]
        records.append(res)
    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean SI-SDR / SI-SDRi grouped by speaker count -- the grading table."""
    agg = (df.groupby("n_src")
             .agg(n=("si_sdri", "size"),
                  si_sdr=("si_sdr", "mean"),
                  si_sdr_input=("si_sdr_mixture", "mean"),
                  si_sdri=("si_sdri", "mean"))
             .round(2)
             .reset_index())
    return agg


# --------------------------------------------------------------------------- #
# CLI: evaluate against a LibriMix-style metadata CSV
# --------------------------------------------------------------------------- #
def _rows_from_librimix(metadata_csv: str, data_root: str, est_dir: str,
                        n_src: int) -> list[dict]:
    """
    Build manifest rows from a LibriMix metadata csv.
    Assumes estimates were written to  <est_dir>/<mixture_id>_sK.wav  (K=1..n_src).
    Paths inside LibriMix metadata are relative to `data_root`.
    """
    md = pd.read_csv(metadata_csv)
    rows = []
    for _, m in md.iterrows():
        mid = m["mixture_ID"]
        ref_paths = [os.path.join(data_root, m[f"source_{k}_path"]) for k in range(1, n_src + 1)]
        est_paths = [os.path.join(est_dir, f"{mid}_s{k}.wav") for k in range(1, n_src + 1)]
        if not all(os.path.exists(p) for p in est_paths):
            continue  # skip utterances the model hasn't produced yet
        rows.append({
            "mixture_id": mid,
            "mixture_path": os.path.join(data_root, m["mixture_path"]),
            "ref_paths": ref_paths,
            "est_paths": est_paths,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="SI-SDRi evaluation harness")
    ap.add_argument("--metadata", required=True, help="LibriMix metadata csv")
    ap.add_argument("--data-root", required=True, help="dir the metadata paths are relative to")
    ap.add_argument("--est-dir", required=True, help="dir with <id>_sK.wav estimates")
    ap.add_argument("--n-src", type=int, required=True)
    ap.add_argument("--out", default=None, help="optional csv to save per-utterance results")
    args = ap.parse_args()

    rows = _rows_from_librimix(args.metadata, args.data_root, args.est_dir, args.n_src)
    if not rows:
        raise SystemExit("No matching estimates found -- run separate.py first.")
    df = evaluate_manifest(rows)
    if args.out:
        df.to_csv(args.out, index=False)
    print(f"\nEvaluated {len(df)} mixtures.\n")
    print(summarize(df).to_string(index=False))


if __name__ == "__main__":
    main()
