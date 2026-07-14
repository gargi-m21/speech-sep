# Person A — Plain-Language Guide (for understanding + explaining to a reviewer)

## 0. The project in one paragraph

We are given a recording where several people talk **at the same time**. The goal is to
split that single recording into **separate clean recordings, one per person**. This is
called **speech separation**. We are doing the **audio-only** version (no video). You
(Person A) build the part that (a) provides the practice/test data, (b) runs ready-made
models that do the splitting, and (c) measures how good any split is.

---

## 1. Audio words

- **Waveform** — audio in a computer is just a long list of numbers. Each number is how
  much the air was pushing (the loudness) at one tiny instant. That list is the waveform.
- **Sample** — one number in that list.
- **Sample rate** — how many samples represent one second. **8 kHz = 8000 numbers per
  second.** All our models were built for 8 kHz, so we convert everything to 8 kHz.
- **Resample** — change the sample rate (e.g. 16000 → 8000 numbers per second) without
  changing how it sounds.
- **Mono / channel** — mono = one waveform (one microphone). We use mono because the task
  is "single-channel" separation.
- **Source** — one person's clean voice, alone.
- **Mixture** — the sources added together into one recording (what you hear when everyone
  talks at once). The model's input is the mixture; its job is to output each source.
- **Speaker / "spk"** — a person talking. **2-spk = a mixture of 2 people talking at once;
  3-spk = 3 people.** The grader tests you at increasing speaker counts (2, then 3, ...).
- **Gain** — a single multiplier applied to a source before mixing, to set its loudness.
  `gain = 0.8` makes a source quieter, `1.1` louder. A clean mixture is literally
  `gain1*source1 + gain2*source2 + gain3*source3`. The gains are stored in the metadata,
  which is why we can rebuild mixtures ourselves.
- **Metadata (CSV)** — a spreadsheet that lists, for each mixture: which source files were
  used and each one's gain. (`.csv` = a plain-text spreadsheet.)

---

## 2. The evaluation metrics (know these cold for the reviewer)

### Decibel (dB)
A unit for ratios, on a "log" scale, used because separation quality covers a huge range.
Rough feel: **+10 dB ≈ 10× better, +20 dB ≈ 100× better.** Higher = better.

### SI-SDR — Scale-Invariant Signal-to-Distortion Ratio
**What it answers:** "How much of my estimated voice is the *real* voice, versus junk?" —
as one number in dB. Higher is better.

- **Signal** = the part of your estimate that truly is the target voice.
- **Distortion** = everything else in your estimate (leftover other speakers, artifacts).
- **Scale-invariant** = it ignores overall volume. A perfect voice that's twice as loud is
  still a perfect separation; you shouldn't lose points for volume. SI-SDR removes volume
  from the comparison.

**The exact steps (this is what `evaluate.py:si_sdr` does):**
1. Center both signals: subtract each one's average so they sit around zero.
2. Find the number `alpha` so that `alpha * (true source)` best matches your estimate.
   `alpha = <estimate, source> / <source, source>`. Call `target = alpha * source`.
   (This step is what makes it scale-invariant — it volume-matches the true voice to you.)
3. `error = estimate - target` — the part of your estimate that is NOT the true voice.
4. `SI-SDR = 10 * log10( energy(target) / energy(error) )`.
   - tiny error → huge ratio → high dB (great separation)
   - error as big as the target → ratio ≈ 1 → ≈ 0 dB (useless)

**Typical values:** raw mixture ≈ 0 dB; good separation ≈ 10–20 dB; our "perfect" sanity
test hit ~99 dB (capped only because we add a microscopic number to avoid dividing by zero).

### SI-SDRi — SI-SDR **improvement**
`SI-SDRi = SI-SDR(your estimate) − SI-SDR(doing nothing)`.
"Doing nothing" = using the raw mixture itself as the guess for each voice (it already
overlaps the true voice a little, so it scores a small baseline). SI-SDRi shows how much
the model **improved over not separating at all**. This is the fair, standard number we
report, per speaker count. A do-nothing baseline gives ≈ 0 (we verified this).

### PIT — Permutation-Invariant matching
**The problem:** the model outputs several voices but doesn't label them. It might output
`[voiceA, voiceB]`. We have true sources `[source1, source2]`. Which output is which? The
model has no fixed order — its output 1 might actually be source 2. If we blindly compared
output1↔source1 and the model used the opposite order, we'd score a *perfect* separation as
terrible.

**The fix:** try **every possible pairing** (permutation) of outputs to true sources, score
each, and keep the best. 2 speakers → 2 pairings; 3 → 6; N → N-factorial (fine for N ≤ 5).

- **Permutation** = an ordering/arrangement.
- **Invariant** = the score does not depend on output order, because we pick the best pairing.

We proved it: we fed the sources in the **wrong** order as fake estimates; PIT tried both
orderings, found the swap, and still scored 99 dB (winning pairing `(1,0)` = "estimate 2
matches source 1, estimate 1 matches source 2"). The same idea is used during *training*
(that's the "T" in PIT) so a model isn't punished for output order; we use it for scoring.

---

## 3. The model words

- **Neural network** — a program with millions of internal numbers that is "trained" on
  examples until it can do a task (here: split a mixture into sources).
- **Weights / checkpoint** — those internal numbers, saved to a file after training.
- **Pretrained** — we use someone else's already-trained weights instead of training from
  scratch. Saves days of GPU time. Our floor uses pretrained models.
- **GPU** — the hardware that runs neural networks fast. Your local one is too small (4 GB),
  so heavy runs go on **Kaggle** (free cloud GPUs).
- **SepFormer** — a well-known, high-quality separation model. Comes pretrained in a
  2-speaker and a 3-speaker version. Handles a **fixed** number of speakers.
- **DPRNN** — another separation model design ("Dual-Path RNN"), good at long audio.
- **MultiDecoderDPRNN** — a DPRNN with several output heads (one per speaker count) **plus a
  small classifier that first guesses how many speakers there are**, then uses the matching
  head. So it handles an **unknown** number of speakers by itself. Pretrained, 2–5 speakers.
  This is what gives Person A the "unknown speaker count" ability without training anything.
- **Backend** — in `separate.py`, "backend" just means "which model to run." There's a
  SepFormer backend now; adding a "MultiDecoderDPRNN backend" = adding code so the same
  script can also run that model. It's just a second plug-in option.
- **Transformer** — a popular neural-network building block; SepFormer is built from it.
  (You don't build these; you run pretrained ones.)

---

## 4. The final approach + dataset decision (why we do it this way)

- **Deadline is 4 days**, so we cannot train big models from scratch. Plan: **Person A** uses
  pretrained models to guarantee a working result and builds the scoring tool; **Person B and
  C** build the two "new" models (MTC-Net, SepTDA) for novelty/credit, backed by that floor.
- **Dataset decision:** the full Libri3Mix is ~332 GB, but that's *every* configuration
  stacked together plus a giant training split. We don't need it. A **clean mixture is just
  `sum(gain * source)`**, and the gains + source file names are in the metadata. So we
  **generate** only the small slices we need from tiny LibriSpeech source audio — no 332 GB,
  no WHAM! noise, no SoX tool.

---

## 5. What Person A (you) actually does

1. **Provide data** — generate the test set (done) and the train slices B/C need, host them
   as one **private** Kaggle Dataset (add B and C as collaborators).
2. **Separate** — run pretrained models to split mixtures into one file per speaker.
3. **Score** — measure any model's output with SI-SDRi (the shared team measuring tool).

---

## 6. The files I wrote for you

> (There is no `dataset.py` — the data-building script is `make_libri3mix_test.py`.)

- **`src/evaluate.py`** — the scorer. Computes SI-SDR and SI-SDRi using PIT. Takes model
  outputs + the true sources, prints a table of SI-SDRi per speaker count. Needs no GPU.
- **`src/separate.py`** — the splitter. Loads a pretrained model, splits each mixture, and
  writes the pieces as `<mixtureID>_s1.wav`, `<mixtureID>_s2.wav`, ... (a naming the scorer
  expects, so the two scripts fit together).
- **`src/make_libri3mix_test.py`** — the data builder. Reads a metadata spreadsheet + the
  LibriSpeech source audio, rebuilds each clean mixture (`sum(gain*source)`), writes the
  mixture + each source, and a `manifest_test.csv` telling the other scripts what was made.
  Works for the test split *and* the training splits, for 2 or 3 speakers.
- **`notebooks/kaggle_person_a.ipynb`** — the Kaggle notebook that runs all of the above in
  order on Kaggle's GPU.
- **`README.md`** — how to run everything. **`requirements.txt`** — the software packages
  needed.

---

## 7. What the Kaggle notebook does, cell by cell

1. **Install** the speech software (speechbrain).
2. **Find** your uploaded code files automatically.
3. **Build data** — download a small piece of LibriSpeech + the LibriMix metadata, then
   generate the 3-speaker test mixtures.
4. **Config** — pick the model and speaker count.
5. **Separate** — run the model on every mixture, writing one file per speaker.
6. **Score** — print the SI-SDRi table.
7. **Listen** (optional) — play the mixture and the separated voices.

**Sanity gate:** on the full test set, the pretrained `sepformer-libri3mix` should score
**≈ 19.8 dB SI-SDRi**. Hitting that proves both the scorer and the generated data are
correct — after which every MTC-Net / SepTDA number is trustworthy.

---

## 8. 30-second version to say to a reviewer

"The task is single-channel speech separation: split one recording of overlapping speakers
into one clean track per speaker. I built the evaluation and baseline pipeline. I generate
the test data on the fly instead of downloading 332 GB, because a clean mixture is just a
gain-weighted sum of source clips stored in the LibriMix metadata. I run pretrained
separators (SepFormer for fixed counts, MultiDecoderDPRNN for unknown counts) and score them
with **SI-SDRi** — scale-invariant signal-to-distortion improvement over the input mixture —
using **permutation-invariant** matching so output ordering doesn't affect the score. My
harness is validated: perfect estimates score ~99 dB, doing nothing scores ~0, and the
pretrained model reproduces its published ~19 dB, which confirms the metric is correct."

---

## 9. Follow-up clarifications

- **Is the data time-series? Is `.wav` just numbers?** Yes to both. Audio is a time series
  (values over time). A `.wav` file = a small header (sample rate, channels) + the list of
  sample numbers. The computer reads those numbers to play or process the sound.
- **Why resample 16k→8k? Do SepTDA/MTC-Net need more resampling?** LibriSpeech source is
  16 kHz; the Libri2/3Mix benchmark and SepFormer run at **8 kHz**, so we downsample once. B
  and C train SepTDA/MTC-Net at **8 kHz too** — the whole team standardizes on 8 kHz, so **no
  extra resampling**; the generated dataset is already 8 kHz for everyone.
- **"clean" does NOT mean denoised.** `mix_clean` is a *name*: a mixture with **only speech,
  no added noise** (vs `mix_both` = speech + noise). Still `Σ gainₖ·sourceₖ`; the gains just
  set loudness (can be <1 or >1) — they don't "clean" anything.
- **How the metadata saves 332 GB.** A mixture is fully described by *which source files* +
  *what gains* — a few bytes of text. So instead of 332 GB of pre-made mixtures, we download
  the small source clips (~6 GB) + tiny metadata and recreate mixtures ourselves. Same result.
- **Is a decibel a loudness unit?** dB is a unit for **ratios** (10·log10 of A/B). Loudness is
  one use; here it measures a different ratio — signal energy vs error energy (SI-SDR).
- **Supervised learning?** Yes — inputs (mixtures) + known correct answers (true sources);
  SI-SDR compares the model's output to them.
- **"Center the signal"** = subtract the mean of all samples so the average is 0; removes a
  constant offset that would distort the energy math. Same waveform, shifted vertically.
- **Model vs Transformer vs Separator.** *Model* = a trained network. *Transformer* = a
  building-block style (uses "attention"). *Separator* = any model whose job is separation.
  SepFormer/MossFormer are separator models built from transformer blocks.
- **SepTDA vs MTC-Net.** SepTDA = a real published model (paper); C implements + trains it.
  MTC-Net = the team's **own** design (MossFormer + TDA) — nothing pretrained; B builds it.
- **Which model does `separate.py` load?** The **SepFormer** backend now (e.g.
  `speechbrain/sepformer-libri3mix`); MultiDecoderDPRNN (unknown count) added Day 2.
- **SpeechBrain** = a free open-source PyTorch toolkit for speech AI; gives us SepFormer +
  code to run pretrained weights without writing the model ourselves.
- **"Sanity gate"** = a must-pass check before trusting results: a known model on a known set
  should reproduce its published score. Ours just caught a real bug (16-bit clipping of
  estimates) — which is exactly what a gate is for.
