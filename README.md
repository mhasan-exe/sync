# PSL Sign Recognition

Recognizes Pakistan Sign Language (PSL) letters and words from a webcam.
One script (`psl.py`) handles recording data, training, live testing, and an optional API.

**How it works:** MediaPipe Holistic extracts hand and upper-body landmarks from each frame. A small GRU model classifies a 30-frame clip of those landmarks. It runs on a normal CPU, with no GPU needed.

- Letters (static) and words (with motion) use the same model.
- Hand positions are stored relative to the shoulders, so body-relative signs work wherever you stand.
- Training augmentation (rotation, scale, noise) helps with tilted and rotated hands.

---

## Setup (Windows)

Use Python 3.10, 3.11 or 3.12. MediaPipe 0.10.21 does not support Python 3.13.

```
py -3.12 -m venv psl_env
psl_env\Scripts\activate
pip install mediapipe==0.10.21 opencv-python "numpy<2" torch fastapi "uvicorn[standard]"
```

Newer MediaPipe versions removed the `mp.solutions` API that this script uses, so keep the version pinned.

---

## Usage

### 1. Record data

```
python psl.py record --label alif --signer m
```

- A camera window opens. Press **SPACE** to record one 30-frame clip. Press **q** to quit.
- Run the command once per sign, changing `--label` each time.
- Use `--signer` for the name of the person signing, so training can validate on a signer it has not seen.
- Clips are saved to `data/<label>/<label>__<signer>__<time>.npy`.

For letters, hold the pose for the whole clip. For words, perform the full motion inside the clip.

### 2. Train

```
python psl.py train
```

- Trains on everything in `data/`.
- Validates on the last signer alphabetically. With a single signer it warns that the accuracy is inflated.
- Keeps the best epoch (not the last) and writes `psl_model.pt` and `labels.json`.

### 3. Test live

```
python psl.py live
```

Shows a box and skeleton on each hand with the prediction in large outlined text (green = confident, yellow = unsure). Press **q** to quit.

### 4. Web app + API

```
pip install "uvicorn[standard]"
python psl.py api
```

Open **http://localhost:8000** and press *Start camera*. The browser sends small frames to the server, which runs the same MediaPipe + model as `live`, and draws the hand box, skeleton, big prediction text and top-3 guesses. Predictions are smoothed over several frames so the label does not flicker.

- To use it from a phone or share a link, the camera needs HTTPS. Install `cloudflared` and run `cloudflared tunnel --url http://localhost:8000` in a second window.
- `POST /predict` still exists for apps: send `{"seq": [[...140 numbers...], ... 30 frames]}` and get `{"sign": "alif", "confidence": 0.93}`.

---

## Recording tips

- Record **30 to 50 clips per sign**, varying distance, angle, lighting and hand tilt.
- Include the **back of the hand** and unusual angles on purpose. The model only learns what you show it.
- Use **2 to 3 different signers** if you can. Number of signers matters more than number of epochs.
- Add a `nothing` label with idle hands and random movement, so the model does not guess when you are not signing.
- Hands are not mirrored in the data. Record some left-handed clips if you want left-hand support.
- Start small (about 10 letters and 5 words), check that the whole pipeline works, then scale up.

## Labels and languages

PSL is one sign language, so there is one model with one label per sign. Use roman Urdu names for letters (`alif`, `bay`, `pay`, ...) and a gloss ID for words (for example `shukriya`).

To show output in Urdu, Pashto, Sindhi, Memoni and English, keep a lookup table from label to text:

```json
{ "shukriya": { "ur": "...", "ps": "...", "sd": "...", "mem": "...", "en": "Thank you" } }
```

Adding a language later only means adding a column. No retraining is needed.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | `pip install mediapipe==0.10.21` |
| pip cannot find that MediaPipe version | You are likely on Python 3.13. Use 3.11 or 3.12 in a venv. |
| NumPy-related errors after installing | `pip install "numpy<2"` |
| `usage: psl.py ... mode is required` | Put the mode first: `python psl.py record --label alif --signer m` |
| Camera window does not open | Close other apps using the camera, or try changing `VideoCapture(0)` to `VideoCapture(1)` |
| Many wrong predictions | Record more clips with more variety and more signers, then retrain |

## Files

| File | Purpose |
|---|---|
| `psl.py` | Everything: record, train, live, api |
| `data/` | Recorded landmark clips (created automatically) |
| `psl_model.pt` | Trained model (created by `train`) |
| `labels.json` | Label list matching the model (created by `train`) |
