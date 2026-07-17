# MTC-Net Speech Separation Project Document

## 1. What This Project Does
This project splits overlapping speech recordings into separate audio tracks. If two or three people are talking at the same time, the model separates the recording so you can listen to each person individually.

---

## 2. Model Architecture (How it Works)
MTC-Net uses a "count-first, mask-later" process. Instead of assuming a fixed number of speakers, it first counts how many people are speaking and then creates the separation masks. 

```
[ Mixed Wav File ] ──► Encoder ──► MossFormer Backbone ──► TDA Count Head (Counts speakers) ──► Masking Head (Creates stencils) ──► Decoder ──► Separate WAVs
```

* **Encoder**: Converts the raw audio (a long list of sound wave numbers) into a 2D digital feature map.
* **MossFormer Backbone**: Cleans up the features. It uses global self-attention to track speaker voices over the whole file, combined with local convolutions to capture fast speech details. It uses weight sharing (recursively running the same block 16 times) to keep the model size under 9 MB.
* **TDA Counting Head**: Computes speaker queries and classifies whether each speaker slot is active.
* **Masking Head**: Generates customized stencils for each active speaker. If the counting head detects 2 speakers, it only runs 2 masking slots. If it detects 3 speakers, it runs 3.
* **Decoder**: Takes the stenciled feature maps and reconstructs them back into independent, playable audio files.

---

## 3. The Counting Problem and Our Workaround
During training, we hit a limitation:
* The counting head can only learn to count if it is trained on mixtures with different numbers of speakers.
* Because of Kaggle storage and time limits, our 2-speaker model was trained only on 2-speaker files (MiniLibriMix), and our 3-speaker model was trained only on 3-speaker files (Libri3Mix).
* As a result, the models overfitted. The 3-speaker model always expects 3 speakers and over-separates 2-speaker files, splitting the voices across 3 tracks and causing leakage.

**Our Solution:**
We built a model selector directly into the web app. The user can select the target speaker count from a dropdown in the UI. The backend api then routes the audio to the correct model checkpoint (2-speaker or 3-speaker), avoiding over-separation.

---

## 4. Results and Performance

### SI-SDRi Scores
SI-SDRi measures how much the separated audio improved compared to the original mixture. Higher decibels (dB) mean better separation.

* **MiniLibriMix (2 Speakers)**: ~11 dB SI-SDRi
* **Libri3Mix (3 Speakers)**: ~14 dB SI-SDRi

[INSERT SCREENSHOT: Plot/Table of SI-SDRi evaluation metrics here]

---

## 5. Web Interface and Demo

[INSERT SCREENSHOT: Uploading file, pipeline loader status, and separated speaker waveforms here]

The frontend is a Next.js web application with a glassmorphism interface. Users upload a WAV file, select the speaker count, and get interactive waveform players with play/pause and download buttons.

---

## 6. How to Run the Demo

### Run the Backend (Python)
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server from the root directory:
   ```bash
   python -m backend.app
   ```

### Run the Frontend (Node.js)
1. Go to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Run the development server:
   ```bash
   npm run dev
   ```
4. Open your browser and go to `http://localhost:3000`.
