# PulseAI

## ECG Heart-Rhythm Anomaly Detection Using Machine Learning

PulseAI is a college-level research prototype for exploring ECG signal processing, heartbeat classification, and transparent machine-learning evaluation. The platform combines a React dashboard, an Express API, a FastAPI inference service, and a TensorFlow/Keras training pipeline based on the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/).

> [!WARNING]
> **Research and education only.** PulseAI is not a medical device, has not been clinically validated, and must not be used to diagnose conditions or make treatment decisions. Uploaded data, model outputs, and generated reports require qualified human review.

## Contents

- [Project goals](#project-goals)
- [Features](#features)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Run locally](#run-locally)
- [Train the CNN](#train-the-cnn)
- [Supported uploads and reports](#supported-uploads-and-reports)
- [API reference](#api-reference)
- [Evaluation and limitations](#evaluation-and-limitations)
- [Project structure](#project-structure)
- [Future work](#future-work)

## Project goals

Manually reviewing long ECG recordings is time-consuming and requires specialized expertise. PulseAI demonstrates how a small, record-separated machine-learning experiment can support:

1. Interactive ECG waveform exploration.
2. Beat-window preprocessing and classification with a 1D CNN.
3. Review flags for non-normal predicted segments.
4. Reproducible evaluation using held-out records.
5. Clear separation between measured results, illustrative UI data, and unavailable model artifacts.

## Features

- Government-inspired navy, white, and teal research dashboard.
- Animated ECG pulse visualization on the main dashboard.
- CSV upload for numeric ECG signal inference.
- PDF, PNG, JPG/JPEG, and WEBP uploads as research attachments.
- Patient name or study ID field for report generation.
- Direct PDF report download with timestamp, file details, safety notice, and embedded image previews.
- SQLite-backed analysis history.
- SQLite-backed registry for the configured 18-record MIT-BIH split.
- Model evaluation view for accuracy, macro precision, macro recall, macro F1, and model status.
- Explicit “model not trained” and “service unavailable” states instead of fabricated predictions.

## Architecture

```text
PulseAI/
├── frontend/       React + Vite dashboard
├── backend/        Express API, upload handling, and SQLite history
├── ml-service/     FastAPI inference API and preprocessing
├── training/       MIT-BIH preparation and TensorFlow/Keras training
└── docs/           Project documentation and report outline
```

### Request flow

```text
Browser
  │
  ├── React dashboard ───────────────┐
  │                                  │
  └── CSV/PDF/image upload           ▼
                              Express backend
                               │         │
                               │         └── SQLite history and training registry
                               ▼
                         FastAPI ML service
                               │
                               └── TensorFlow/Keras model
```

## Technology stack

| Layer | Technologies |
| --- | --- |
| Frontend | React 18, Vite, Recharts, Lucide React, jsPDF |
| Backend | Node.js, Express, Multer, better-sqlite3 |
| ML service | Python, FastAPI, NumPy, SciPy |
| Training | TensorFlow/Keras, WFDB, scikit-learn, Matplotlib |
| Dataset | MIT-BIH Arrhythmia Database via PhysioNet |

## Prerequisites

- Node.js 20 or newer.
- Python 3.10 or 3.11 for training and inference dependencies.
- Google Colab GPU recommended for the training experiment.
- Internet access when downloading MIT-BIH records through WFDB.

> **Python note:** TensorFlow dependencies are not guaranteed to support every newer Python release. Use Python 3.10/3.11 or Google Colab for the documented training workflow.

## Run locally

PulseAI uses three local services. Open three terminal windows from the project root.

### 1. Start the ML service

```powershell
cd ml-service
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8001
```

For macOS/Linux:

```bash
source .venv/bin/activate
```

Without a trained model, `GET /health` reports `model_loaded: false`. CSV inference remains disabled and returns an explicit error.

### 2. Start the backend

```powershell
cd backend
npm install
npm run dev
```

The API runs at <http://localhost:8000>.

### 3. Start the frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally <http://localhost:5173>.

## Train the CNN

The repository intentionally does not include trained model weights. Train the educational baseline in Google Colab or a compatible Python 3.10/3.11 environment.

### Install training dependencies

```bash
cd training
pip install -r requirements.txt
```

### Run the default experiment

The default configuration uses 18 records:

- 12 training records
- 6 held-out test records
- Record-level separation to reduce leakage
- Fixed random seed for reproducibility

```bash
python train_mitbih_cnn.py
```

### Customize the record split

Keep the total number of records between 15 and 20 for this college-level experiment. Never overlap training and test records.

```bash
python train_mitbih_cnn.py `
  --train-records 100 101 103 105 106 108 109 111 112 `
  --test-records 102 104 107 117 118 119
```

The script downloads the selected records from PhysioNet and creates:

```text
training/artifacts/
├── best_model.keras
├── ecg_cnn.keras
├── label_map.json
├── metrics.json
├── classification_report.txt
├── confusion_matrix.png
└── training_history.png
```

Copy the runtime artifacts to `ml-service/models/`:

```text
ecg_cnn.keras
label_map.json
metrics.json
```

Then restart the ML service. The dashboard will read measured metrics when the artifacts are available. The backend also reads `training/artifacts/metrics.json` for evaluation display.

## Supported uploads and reports

### CSV signal analysis

CSV files must contain a single numeric ECG channel with at least 100 samples. A header row is allowed if non-numeric values are skipped. The current inference service normalizes fixed-size windows before passing them to the CNN.

CSV classification requires a trained model artifact.

### PDF and image attachments

The following formats are accepted:

- PDF
- PNG
- JPG/JPEG
- WEBP

These files are stored as research attachments in the local analysis history. They are **not** classified by the current CNN because the model expects numeric ECG samples. The application does not invent predictions from images or documents.

### PDF reports

After an upload:

1. Enter a patient name or study ID.
2. Choose **Generate report** for the browser report view, or **Download PDF report** for a direct PDF file.
3. Review the timestamp, source file, analysis status, safety notice, and any embedded image preview.

Reports are research documents, not clinical reports. Do not include identifiable patient data unless you have appropriate authorization and secure handling procedures.

## API reference

### Express backend — `http://localhost:8000`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Backend and ML-service status |
| `GET` | `/api/history` | Recent saved uploads and analyses |
| `GET` | `/api/metrics` | Evaluation metrics when an artifact is available |
| `GET` | `/api/training-data` | Configured MIT-BIH records and artifact status |
| `POST` | `/api/analyze` | Multipart upload using the `ecg` field |

### FastAPI ML service — `http://localhost:8001`

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Service availability and model-load state |
| `POST` | `/predict` | Segment and classify a numeric ECG signal |

Example prediction request:

```json
{
  "signal": [0.12, 0.14, 0.11, 0.09]
}
```

The real request must contain at least 100 samples.

## Evaluation and limitations

The training pipeline reports:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Classification report
- Confusion matrix

Important limitations:

- The default experiment uses only 18 records.
- MIT-BIH class frequencies are imbalanced.
- The simplified label mapping is an educational baseline, not a complete clinical standard.
- The current web segmenter uses fixed windows rather than a validated clinical R-peak detector.
- Image and PDF uploads are not converted into ECG waveforms.
- The model has no clinical validation and must not be used for patient care.

## Project structure

```text
backend/
├── package.json
└── server.js

frontend/
├── package.json
├── vite.config.js
└── src/
    ├── App.jsx
    ├── main.jsx
    └── styles.css

ml-service/
├── app.py
├── requirements.txt
└── models/

training/
├── train_mitbih_cnn.py
├── requirements.txt
└── PulseAI_Training_Colab_Starter.ipynb
```

## Future work

- Add WFDB record upload and annotation-aware evaluation.
- Add authenticated, encrypted report storage for approved research use.
- Add multi-lead ECG support and sampling-rate metadata.
- Compare the baseline CNN with CNN-LSTM and other architectures.
- Add batch report generation and richer per-class visualizations.
- Add automated tests for upload validation, report generation, and model responses.

## Dataset and references

- [MIT-BIH Arrhythmia Database — PhysioNet](https://physionet.org/content/mitdb/1.0.0/)
- [PhysioNet](https://physionet.org/)
- [WFDB Python package](https://wfdb.readthedocs.io/)

Review the dataset’s access, citation, and usage terms before using it in a report or presentation.

## License

This repository is a college-level research project. Add the project’s chosen license and a `LICENSE` file before distributing it publicly.
