# Speech Separation — FINAL Plan (4 working days, team of 3)

## Context

**Assignment:** single-channel recording with 3+ overlapping speakers → one clean waveform per
speaker (monaural speech separation). Graded on **(a) increasing concurrent-speaker counts** and
**(b) separation quality**. Audio-only (video/speaker-ID not required). Not RAG/LLM — this is DSP +
deep learning.

**Hard constraints (drive every decision below):**
- **Deadline 15 Jul 2026 → 4 working days (11–14 Jul), submit 15th.**
- **GPU: 90 hrs on Kaggle (team total). Local GPU = 4 GB dedicated → DEV/DEBUG ONLY, no training.**
- Team of 3.

**Consequence:** everything trains and heavy-runs **on Kaggle (Linux)** — which also removes the
Windows wget/SoX/schannel problems already hit locally. Target a **working trained system**, not a
from-scratch SOTA reproduction.

## Decisions (recorded — reviewer-facing rationale)

| Decision | Choice | Why |
|---|---|---|
| Modality | Audio-only, single-channel | Brief says video/speaker-ID not required |
| Speaker count | Unknown / variable | Matches the multi-speaker-level grading |
| Guaranteed floor | Pretrained pipeline + SI-SDRi eval | A valid, working submission by Day 2 regardless of training outcomes |
| Novelty (primary) | **MTC-Net** = pretrained backbone + **TDA** | GPU-efficient (backbone frozen); genuine unknown-count novelty |
| Baseline (compare) | **SepTDA**, trained at **reduced scale** | Strong published unknown-count model; shared TDA code with MTC-Net |
| Dropped | MAG-Net | Too risky for 4 days (Mamba + 6 novel parts) |
| Train env | **Kaggle only** (Linux GPUs) | 90 hr budget; local 4 GB can't train; avoids Windows toolchain issues |
| Metric | **SI-SDRi**, per speaker-count level (2/3/4/5) | Standard metric; mirrors grading |
| Data | Kaggle-hosted LibriMix subset (+ MiniLibriMix for local debug) | Generate/pull on Kaggle; free; Libri3Mix gives 3-spk |

## Shared foundation (build once, all tracks reuse)

- **TDA module (Transformer Decoder Attractor):** learned speaker queries → transformer decoder →
  one attractor per speaker + a **stop/existence probability** → unknown count. **Used by BOTH
  MTC-Net and SepTDA** — one person implements it cleanly, both import it.
- **Input→output pipeline:** load mixture → (resample 8k mono) → model → write `s1..sN.wav`.
- **SI-SDRi eval harness:** reuse `asteroid.losses.PITLossWrapper` + `pairwise_neg_sisdr` (or
  `torchmetrics` SI-SDR). **Do not hand-roll PIT.** Report SI-SDRi per speaker count.

## Three tracks / team split

- **Person A — Floor + Eval + Data.** Pretrained pipeline (`speechbrain/sepformer-libri2mix`
  /`-libri3mix`; `JunzheJosephZhu/MultiDecoderDPRNN` for unknown count). SI-SDRi harness. LibriMix
  data hosted as a Kaggle Dataset. **Deliverable = the guaranteed submission.**
- **Person B — MTC-Net (novelty).** Backbone = **MossFormer2** (pretrained, ClearerVoice-Studio /
  ModelScope) **— if its weights don't load cleanly on Day 1, fall back to pretrained SepFormer**
  (guaranteed available in SpeechBrain). Bolt on the shared TDA head; FiLM-condition the mixture
  embedding on each attractor → per-speaker masks. Loss = PIT SI-SDR + BCE on the existence head.
  **Freeze backbone, train the attractor/conditioning**, then optionally unfreeze to fine-tune.
- **Person C — SepTDA (baseline).** Implement SepTDA (dual-path → TDA → triple-path) using the
  shared TDA module; look for official/community code first. **Train at reduced scale** (Libri2Mix
  or WSJ0-2mix, smaller model / fewer epochs) to a *working* checkpoint; cite the paper's published
  numbers as the SOTA reference. If GPU/time runs short → partial training + published-number
  comparison.

## GPU budget (~90 hrs, keep ~10 hr buffer)

- MTC-Net (frozen backbone): ~20–30 hr · SepTDA (reduced): ~30–40 hr · eval/inference: ~10 hr ·
  debugging: ~10 hr. **Checkpoint every epoch to Kaggle output** (sessions are time-limited).

## Data

- **MiniLibriMix** (already downloaded, 2-spk) — local sanity/debug only.
- **On Kaggle:** search for an existing LibriMix/WSJ0-mix **Kaggle Dataset** first (skip generation).
  Else generate **Libri2Mix + Libri3Mix at 8k / min / mix_clean** there (Linux → the generator's
  wget/SoX just work) and save as a Kaggle Dataset attached to all notebooks.
- **4/5-speaker test sets** (for the higher grading levels): generate small **test-only** mixtures by
  summing N random LibriSpeech utterances. `MultiDecoderDPRNN` (pretrained 2–5 spk) is the
  unknown-count baseline to beat.

## Day-by-day (11–14 Jul, submit 15th)

- **Day 1 (11th) — Setup + first win.** Kaggle envs (`speechbrain`, `asteroid`, `torchaudio`).
  A: `sepformer-libri3mix` separating one mixture end-to-end + host data. B: load MossFormer2 (or
  confirm SepFormer fallback). C: locate SepTDA code, draft the **shared TDA module**. Eval skeleton.
- **Day 2 (12th) — Floor secured.** A: MultiDecoderDPRNN unknown-count + SI-SDRi for 2/3-spk →
  **submission now guaranteed.** B: MTC-Net forward pass, start training attractor head. C: SepTDA
  forward pass, start reduced-scale training.
- **Day 3 (13th) — Train + measure.** B: MTC-Net unknown-count numbers. C: SepTDA converging. A:
  extend eval to 4/5-spk + comparison-table scaffold.
- **Day 4 (14th) — Finalize.** Collect SI-SDRi across speaker counts for all models; pick submission
  model(s); build demo (CLI or small Gradio app); write report; buffer.

## Verification

1. **Smoke test:** pretrained separator on a `mix_clean` file → distinct audible outputs (compare to
   `s1`/`s2` targets by ear).
2. **Harness sanity:** `sepformer-libri3mix` ≈ **19.8 dB SI-SDRi** on Libri3Mix test → confirms the
   metric before trusting novelty numbers.
3. **Grading dry-run:** eval across 2/3/4/5-spk → the per-level SI-SDRi table evaluators reproduce.
4. **Novelty check:** MTC-Net predicts correct speaker count + report SI-SDRi vs SepTDA and
   MultiDecoderDPRNN baselines.

## Reliability guardrails (the "grounded" part)

- **The grade never depends on training success** — Tier-0 pretrained pipeline is a complete,
  working submission on its own.
- **MTC-Net has a backbone fallback** (SepFormer) if MossFormer2 weights are troublesome.
- **SepTDA has a scope fallback** (reduced training / published-number comparison) if GPU/time tight.
- **All heavy compute on Kaggle**; local 4 GB GPU only edits/debugs.