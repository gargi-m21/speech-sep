# Speech Separation — Person A (Floor + Evaluation)

The **guaranteed-working pipeline** + the **SI-SDRi evaluation harness** that scores
every model on the team (pretrained baselines, MTC-Net, SepTDA) the same way.

## What's here

| File | Role |
|---|---|
| `src/evaluate.py` | SI-SDRi harness (PIT matching). The team's "ruler". CPU-only, no model needed. |
| `src/separate.py` | Run a pretrained separator → one wav per speaker. Writes `<id>_sK.wav`. |
| `src/make_libri3mix_test.py` | Build the Libri3Mix **test** set from LibriSpeech test-clean only (no WHAM!/SoX/332 GB). Emits a manifest the other two files consume. |
| `notebooks/kaggle_person_a.ipynb` | Runnable Kaggle notebook: build test set → separate → score. |
| `data/MiniLibriMix/` | Tiny 2-speaker set (already extracted) for local debugging. |

## Why we generate instead of download

The full Libri3Mix is ~332 GB — but that's almost all *training* data. Evaluation needs
only the **test** split (LibriSpeech test-clean, ~350 MB). A clean mixture is just
`Σ gainₖ·sourceₖ`, and the gains live in the LibriMix metadata, so
`make_libri3mix_test.py` rebuilds the test set with no WHAM!, no SoX, no shell script.

## The two concepts (be able to explain these)

- **SI-SDR** — how clean a recovered voice is, in dB, *independent of loudness*
  (scale-invariant). **SI-SDRi** = improvement over feeding the raw mixture.
- **PIT** — the model outputs voices in arbitrary order; we try every ordering and
  keep the best match so correct separations aren't punished for ordering.

## Local (CPU) — validate the ruler

Evaluation needs no GPU. This is how the harness was verified:

```bash
python - <<'PY'
import sys; sys.path.insert(0, "src")
import os, pandas as pd, evaluate as E
md = pd.read_csv("data/MiniLibriMix/metadata/mixture_train_mix_clean.csv").head(20)
rows=[]
for _,m in md.iterrows():
    rows.append(dict(
        mixture_id=m["mixture_ID"],
        mixture_path=os.path.join("data", m["mixture_path"]),
        ref_paths=[os.path.join("data", m["source_1_path"]), os.path.join("data", m["source_2_path"])],
        # perfect estimates == the references → sanity check (expect ~99 dB)
        est_paths=[os.path.join("data", m["source_1_path"]), os.path.join("data", m["source_2_path"])],
    ))
df = E.evaluate_manifest(rows)
print(E.summarize(df).to_string(index=False))
PY
```

Expected sanity results: perfect estimates ≈ **99 dB**, do-nothing ≈ **0 dB SI-SDRi**.

## Kaggle — the real run (GPU)

Local can't train and struggles to run the models (4 GB / CPU-only torch), so do
model runs on Kaggle:

1. New Kaggle Notebook → **Settings → Accelerator: GPU (T4/P100)**, Internet: **On**.
2. Upload `src/separate.py` and `src/evaluate.py` (or paste them), plus attach a
   **LibriMix** dataset (search Kaggle for an existing Libri2Mix/Libri3Mix, else
   generate — see the plan's Data section).
3. Run `notebooks/kaggle_person_a.ipynb`. It: installs speechbrain, separates the
   test set with a pretrained SepFormer, then scores it with `evaluate.py`.

### Sanity gate before trusting anything

`speechbrain/sepformer-libri3mix` on the Libri3Mix test set must score
**≈ 19.8 dB SI-SDRi**. If it doesn't, the harness or the data wiring is wrong —
fix that *before* reporting any MTC-Net / SepTDA numbers.

## Interfaces the team relies on

- Estimates are written as `<outdir>/<mixture_id>_sK.wav`, K = 1..N.
- To score any model: point `evaluate.py --est-dir` at that folder + the matching
  LibriMix `--metadata` / `--data-root` / `--n-src`. Works identically for
  MTC-Net and SepTDA outputs.

---

## 🌐 Running the MTC-Net Web Application

We integrated a premium Web App featuring:
- **FastAPI backend** that loads checkpoints and exposes a `/separate` REST API.
- **Next.js frontend** designed with a dark glassmorphism theme, interactive architecture flow, and `wavesurfer.js` audio players.

### Prerequisites
- Python 3.8+
- Node.js 18+ and npm

### Step 1: Install Dependencies
1. **Python Dependencies** (install from root directory):
   ```bash
   pip install -r requirements.txt
   ```
2. **Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Step 2: Start the FastAPI Backend
Start the FastAPI server from the repository root:
```bash
python -m uvicorn backend.app:app --reload --port 8000
```
* The API will load the checkpoints, map MossFormer model keys on the fly, and start on [http://localhost:8000](http://localhost:8000).
* You can view the API Swagger docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Step 3: Start the Next.js Frontend
Open a new terminal window, navigate to the `frontend/` directory, and start the development server:
```bash
cd frontend
npm run dev
```
* The frontend will spin up on [http://localhost:3000](http://localhost:3000).
* Open the browser and visit [http://localhost:3000](http://localhost:3000) to upload audio and visually verify separate speaker tracks.

### 🧪 Verification / Testing Scripts
We created audit and integration scripts in the artifacts directories to inspect the internal model states:
* **`python scratch/test_api_separation.py`**: Spin up the backend, feed it synthetic wave mixtures, and verify response formats and routing logic.
* **`python scratch/audit_tda.py`**: Audit intermediate tensor shapes and stats of TDA attractors and masks.
* **`python scratch/audit_predicted_count.py`**: Perform count propagation diagnostics from TDA to the decoder.

