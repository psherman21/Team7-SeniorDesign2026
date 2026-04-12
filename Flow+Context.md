# FLEX-GLOVE-ML — Project Context

**Team 7 Senior Design — University of Pittsburgh, Swanson School of Engineering**
Author: Philip Sherman

---

## What This Project Is

A data glove system that reads flex sensor and IMU data from a hardware glove over serial/Bluetooth and uses a KNN machine learning model to recognize American Sign Language (ASL) hand gestures in real time. The UI displays the predicted letter, confidence score, top 5 predictions, and live sensor readings. A demo mode allows users to spell their name live using the glove.

---

## Hardware

- **5 flex sensors** — one per finger (thumb → pinky). Raw output is a voltage (0–3.3V range). Before any data is saved or used for inference, the values are **calibrated to a 0–100 scale** using a per-session open-hand/closed-fist calibration routine.
- **IMU** — provides accelerometer (x/y/z) and gyroscope (x/y/z). Raw values are in ADC counts. The IMU is passed through without normalization.
- **Microcontroller** — communicates over USB serial (default COM6) at 115200 baud.

### Serial Packet Format

The packet format is now **configurable** via `sensor_config.json` (see Sensor Config section). The default is flex-only (5 values per packet). The full format when all fields are enabled is:

```
timestamp,flex1,flex2,flex3,flex4,flex5,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
```

- `timestamp` — milliseconds since boot (stored separately, not used in ML)
- `flex1–5` — raw ADC voltage readings, calibrated to 0–100 before use
- `accel_x/y/z` — raw accelerometer counts
- `gyro_x/y/z` — raw gyroscope counts

**Current firmware status:** Only sends 5 flex values. The packet format is set to flex-only in `sensor_config.json` to match. When the firmware is updated to send IMU data, enable those fields in the ⚙ settings window.

### Calibration

Before recording data or running inference, the user runs a calibration:
1. Open hand fully for 3 seconds → captures `sensor_min`
2. Close hand into a fist for 3 seconds → captures `sensor_max`

Calibration maps raw voltages to 0–100 for flex sensors only. IMU values are passed through unchanged. `calibration_done` flag is set after a successful calibration — the UI warns if you try to save data without calibrating.

---

## Project File Structure

```
FLEX-GLOVE-ML/
├── app/
│   └── main.py                       # Tkinter UI — main application
├── assets/
│   └── pitt_logo.png                 # University of Pittsburgh logo
├── config/
│   (sensor_config.json lives at project root, not here)
├── data/
│   ├── raw/                          # Team-recorded gesture CSVs
│   ├── processed/                    # Merged/cleaned data for training
│   └── external/                     # Imported external dataset
├── hardware/
│   ├── data_logger.py                # Standalone serial recording script
│   ├── BLE_scanner.py                # Bluetooth scanner utility
│   └── find_UUID.py                  # BLE UUID finder
├── ml/
│   ├── train_model.py                # KNN training script
│   ├── gen_dummy_data.py             # Generates fake gesture data for testing
│   └── import_external_gestures.py  # Imports + normalizes external dataset
├── models/
│   ├── knn_model.joblib              # Trained KNN model
│   └── scaler.joblib                 # StandardScaler fitted on training data
├── notebooks/                        # Jupyter notebooks for experimentation
├── ASL-Sensor-Dataglove-Dataset/
│   └── 001/ … 025/                   # External dataset: 25 subjects, single-letter CSVs
├── sensor_config.py                  # Shared packet format utility (project root)
├── sensor_config.json                # Active packet format config (project root)
├── .gitignore
└── README.md
```

---

## Sensor Config System

The packet format is controlled by two files at the project root:

### `sensor_config.json`
```json
{
    "fields": {
        "timestamp": false,
        "flex_1":    true,
        "flex_2":    true,
        "flex_3":    true,
        "flex_4":    true,
        "flex_5":    true,
        "accel_x":   false,
        "accel_y":   false,
        "accel_z":   false,
        "gyro_x":    false,
        "gyro_y":    false,
        "gyro_z":    false
    }
}
```

### `sensor_config.py`
Shared utility module imported by both `main.py` and `data_logger.py`. Key functions:
- `load_config()` — reads JSON, returns field dict
- `save_config(fields)` — writes back to JSON
- `get_active_fields()` — ordered list of enabled field names
- `get_active_sensor_fields()` — same but excludes timestamp
- `get_expected_len()` — total values expected per serial packet
- `get_csv_header()` — column names for CSV output
- `parse_packet(line)` — parses one raw serial line, returns dict or None
- `validate_packet(parsed)` — range-checks all values, returns (bool, warnings)

Both scripts call `sc.parse_packet()` for all serial reading — no hardcoded field positions anywhere. The UI reads the config fresh on every packet so changes take effect immediately after saving in the settings window. `data_logger.py` reads config on startup so requires a restart to pick up changes.

### Changing the format
Open the ⚙ gear button (next to Calibrate in the Connection frame) to open the settings window. Toggle fields on/off and hit Save. A canvas diagram shows the current packet layout in real time as you toggle. The UI updates immediately; restart `data_logger.py` to apply there.

---

## Data Format

### Internal (team-collected) CSV schema

Saved by the UI to `data/{selected_folder}/gesture_{LETTER}.csv`. Columns match whatever sensor fields are active in `sensor_config.json` at the time of recording, plus a `label` column:

```
flex_1, flex_2, flex_3, flex_4, flex_5, [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z,] label
```

- Flex values are **calibrated 0–100** (not raw voltages)
- Label is a single uppercase letter (A–Z)
- Multiple recording sessions for the same letter **append** to the same file — no duplicate files
- Recording duration is set via a text entry box (no upper limit — can record for 30s, 60s, etc.)

### External dataset schema (ASL-Sensor-Dataglove-Dataset)

```
timestamp, user_id, flex_1–5, Qw, Qx, Qy, Qz, GYRx–z, ACCx–z, ACCx_body–z, ACCx_world–z
```

- Flex values are already on a 0–100 scale (pre-normalized by the researchers)
- Contains quaternion orientation + three ACC frames — much richer IMU than the team glove
- `import_external_gestures.py` extracts only: `flex_1–5`, `ACCx/y/z`, `GYRx/y/z` to match internal schema
- Files named with a single letter (e.g. `a.csv`) are treated as gesture labels

---

## ML Pipeline

### Model

**K-Nearest Neighbors (KNN)**
- `k = 5`, `weights = 'distance'`, `metric = 'euclidean'`
- Input: however many sensor features are active in `sensor_config.json` (currently 5 — flex only)
- Output: predicted ASL letter (A–Z)
- Scaler: `StandardScaler` fitted on training data, saved separately
- At inference time, `predict_gesture()` auto-trims input to `scaler.n_features_in_` so the model works regardless of how many fields are active

### Constants (hardcoded — change both together if retraining)

```python
K_NEIGHBORS = 5       # in app/main.py and ml/train_model.py
SMOOTHING_WINDOW = 3  # ~150ms at 16Hz — balances noise vs. latency for live ASL
```

### Training

```bash
python ml/train_model.py --data data/raw/gesture_A.csv
# or for external data:
python ml/train_model.py --data data/external/gesture_A.csv
```

Saves model to `models/knn_model.joblib` and scaler to `models/scaler.joblib`.

### Inference (in UI)

1. Raw serial packet received
2. Parsed via `sc.parse_packet()` — fields determined by `sensor_config.json`
3. Timestamp stored separately if present, not passed to ML
4. Flex values calibrated (0–100)
5. Rolling buffer of last `SMOOTHING_WINDOW` frames averaged
6. Input trimmed to `scaler.n_features_in_` features
7. `scaler.transform()` applied
8. `model.predict()` called
9. Stability filter: prediction only shown if same letter appears ≥ 3 times in last 5 frames
10. `predict_proba()` used to get top 5 confidence scores

---

## UI (`app/main.py`)

Built with **Tkinter**. Window size: ~925×700px.

### Left column (controls)
- **Connection** — COM port entry, Connect/Disconnect toggle, Calibrate, ⚙ sensor settings
- **Recognition Control** — dataset dropdown (scans `data/` subfolders for CSVs), ↻ refresh, Load Model, Start/Stop Recognition, ▶ Demo Mode (enabled only after model is loaded)
- **Confidence Threshold** — slider (0–100%, default 60%). Predictions below threshold show "?" instead of a letter
- **Data Collection** — gesture label entry, recording duration text box (no upper limit, default 10s), Record Gesture, Save to CSV

### Right column (display)
- **Top 5 Predictions** — letter label, progress bar, confidence % for each of top 5
- **Current Recognition** — large letter display, status text, confidence % and bar
- **Demo Mode panel** (hidden until activated) — word display, hold progress bar, Backspace, Clear
- **Live Sensor Values** — flex sensors (voltage + bar), accelerometer x/y/z, gyroscope x/y/z

### Key behaviors
- Connecting to the glove automatically starts the recognition/display loop
- The dataset dropdown controls where **both** recorded CSVs are saved and which folder is shown as context when loading the model
- Saving a gesture appends to `gesture_{LABEL}.csv` in the selected dataset folder — no overwriting, no duplicate files
- The pulsing dot in the top-left turns green when packets are being received and shows live Hz rate
- Sensor settings (⚙) open a Toplevel window — changes take effect immediately in the UI without reconnecting

---

## Demo Mode

Accessed via the **▶ Demo Mode** button in Recognition Control (only enabled after a model is loaded).

### Flow
The user signs ASL letters freely. Each letter is confirmed by holding it steadily for 1.5 seconds. The word builds up in a large text display visible to an audience.

### State machine
```
WATCHING → HOLDING → CONFIRMED → COOLDOWN → WATCHING (loop)
              ↓ (confidence drops or letter changes)
           WATCHING (reset)
```

### Timing constants
```python
DEMO_HOLD_DURATION  = 1.5   # seconds to hold (~24 frames at 16Hz)
DEMO_COOLDOWN       = 1.0   # lockout after confirm (handles double letters)
DEMO_BOLD_FLASH     = 0.5   # seconds new letter stays green after confirm
DEMO_MIN_CONFIDENCE = 60.0  # minimum confidence % to start hold timer
```

### Controls
- **⌫ Backspace** — remove last confirmed letter
- **✕ Clear Word** — wipe entire word and restart
- **■ Exit Demo** — return to normal mode (word is cleared on exit)

---

## Gestures Being Classified

The project is collecting and classifying **ASL fingerspelled letters**:

| Checkoff | Letters |
|----------|---------|
| First    | 1, 2, 3, OK, W |
| Second   | A, B, C, D, E, F, I, L, O, Y |
| Third    | J, M, N, P, S, T, U, V, X, Z |
| Final    | R, K, Q, G, H |

---

## External Dataset

**ASL-Sensor-Dataglove-Dataset** — 25 subjects (folders `001`–`025`), each containing single-letter CSV files. Files named with a single letter (e.g. `a.csv`, `b.csv`) correspond to that ASL letter.

Import script: `ml/import_external_gestures.py`
- Set `EXTERNAL_ROOT = "ASL-Sensor-Dataglove-Dataset"` and `OUTPUT_DIR = "data/external"`
- Combines all 25 subjects' data for each letter into one `gesture_{LETTER}.csv`
- Drops: timestamp, user_id, quaternions, body-frame and world-frame ACC
- Keeps and renames: flex_1–5, ACCx→accel_x, ACCy→accel_y, ACCz→accel_z, GYRx→gyro_x, GYRy→gyro_y, GYRz→gyro_z

---

## Known Issues / TODO

- [ ] Firmware not yet sending full packet with IMU — currently flex-only; enable IMU fields in ⚙ settings once firmware is updated
- [ ] Bluetooth connection not yet implemented (`data_logger.py` has a TODO for this)
- [ ] `data_logger.py` does not use the UI-selected dataset folder — output path is still hardcoded at the top of the script
- [ ] IMU values from external dataset are in different units than team glove — may need normalization before mixing datasets
- [ ] J and Z are motion-based ASL letters (not static poses) — KNN may not classify them reliably

---

## Dependencies

```
pyserial
numpy
pandas
scikit-learn
joblib
tkinter (stdlib)
Pillow
```

Run from the project root (`flex-glove-ml/`) with the venv activated:

```bash
python app/main.py
```