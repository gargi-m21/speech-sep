# MTC-Net Speech Separation Project

This repository contains the implementation of MTC-Net, an end-to-end neural network for speech separation with dynamic speaker counting. It also includes the web backend (FastAPI) and frontend (Next.js) for running interactive audio separation demos.

## Repository Contents

| Directory/File | Role |
|---|---|
| `mtcnet/` | PyTorch implementation of the MTC-Net architecture (Encoder, MossFormer Backbone, TDA Counting Head, Masking Slots, and Decoder). |
| `backend/` | FastAPI REST API that loads checkpoints and processes audio files. |
| `frontend/` | Next.js interactive web interface for uploading files and playing back separated tracks. |
| `src/` | Helper scripts for dataset generation (`make_libri3mix_test.py`) and SI-SDRi evaluation (`evaluate.py`). |
| `train.py` | Training script for MTC-Net. |
| `PROJECT_DOC.md` | Detailed explanation of the architecture, counting limitations, and results. |

---

## 🌐 Running the Web Application Demo

You will need two terminal windows: one for the backend and one for the frontend.

### Step 1: Install Dependencies
Run this command from the project root folder to install PyTorch, FastAPI, and other python libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Start the FastAPI Backend
Start the FastAPI server from the project root folder:
```bash
python -m backend.app
```
This loads the weights for the 2-speaker model (`MiniLibriMix_mtcnet_weights.pth`) and the 3-speaker model (`Libri3Mix_mtcnet_weights.pth`) on CPU or GPU and starts the API server on `http://localhost:8000`.

### Step 3: Start the Next.js Frontend
Open a new terminal window, navigate into the `frontend` folder, install the packages, and run the server:
```bash
cd frontend
npm install
npm run dev
```
Open your browser and visit `http://localhost:3000`. You can upload WAV files, select the target speaker model (2 speakers, 3 speakers, or Auto-Detect), run the separation, and listen to the separated outputs.

---

## 🧪 Testing Locally (SI-SDRi Evaluation)

To verify the evaluation metrics and test the separation logic on the local MiniLibriMix set, run:
```bash
python scratch/test_separation.py
```
This script runs a test separation on a local file from `data/MiniLibriMix/val/mix_clean` and prints out the sorted speaker RMS energies and model details.
