"""
PulseAI baseline training script — MIT-BIH Arrhythmia Database.

Educational baseline, not a clinical model.
IMPORTANT: split by RECORD before extracting beat windows to reduce leakage.
The record list below is deliberately small for a college demo. Increase it for
a defensible experiment and report exact record IDs in your project report.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import wfdb
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WINDOW = 187
# MIT-BIH record-level split. Keep test records separate from training records.
# This intentionally uses a small 18-record subset for a college project.
DEFAULT_TRAIN_RECORDS = ["100", "101", "103", "105", "106", "108", "109", "111", "112", "113", "115", "116"]
DEFAULT_TEST_RECORDS = ["102", "104", "107", "117", "118", "119"]
# AAMI-style simplified mapping: N, V, A, Other.
# This is a teaching baseline and not a full standards-compliant clinical mapping.
LABEL_MAP = {"0": "N", "1": "V", "2": "A", "3": "Other"}
BEAT_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "V": 1, "!": 1,
    "A": 2, "a": 2, "J": 2, "S": 2,
}
OTHER_SYMBOLS = {"F", "/", "f", "Q", "E"}
def class_id(symbol):
    if symbol in BEAT_MAP: return BEAT_MAP[symbol]
    if symbol in OTHER_SYMBOLS: return 3
    return None

def download_records(records):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for rec in sorted(set(records)):
        header = DATA_DIR / f"{rec}.hea"
        if not header.exists():
            print(f"Downloading MIT-BIH record {rec}...")
            wfdb.dl_database("mitdb", str(DATA_DIR), records=[rec])
    print("Dataset files available in", DATA_DIR)

def load_record(rec):
    # Use MLII / first channel, sampling frequency is normally 360 Hz.
    record = wfdb.rdrecord(str(DATA_DIR / rec), channels=[0])
    ann = wfdb.rdann(str(DATA_DIR / rec), "atr")
    signal = record.p_signal[:, 0].astype(np.float32)
    X, y = [], []
    half = WINDOW // 2
    for pos, sym in zip(ann.sample, ann.symbol):
        label = class_id(sym)
        if label is None: continue
        start, end = int(pos) - half, int(pos) + (WINDOW-half)
        if start < 0 or end > len(signal): continue
        beat = signal[start:end]
        beat = beat - beat.mean()
        std = beat.std()
        if std > 1e-7: beat = beat / std
        X.append(beat)
        y.append(label)
    return X, y

def dataset(records):
    xs, ys = [], []
    for rec in records:
        x, y = load_record(rec)
        xs.extend(x); ys.extend(y)
        print(f"Record {rec}: extracted {len(y)} labelled beats")
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)

def build_model():
    inp = tf.keras.Input(shape=(WINDOW, 1))
    x = tf.keras.layers.Conv1D(32, 7, padding="same", activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Conv1D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train the PulseAI ECG CNN on a small MIT-BIH record subset.")
    parser.add_argument(
        "--train-records",
        nargs="+",
        default=DEFAULT_TRAIN_RECORDS,
        help="MIT-BIH record IDs used for training (default: 12 records).",
    )
    parser.add_argument(
        "--test-records",
        nargs="+",
        default=DEFAULT_TEST_RECORDS,
        help="MIT-BIH record IDs held out for testing (default: 6 records).",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    return parser.parse_args()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(42)
    train_records = list(dict.fromkeys(args.train_records))
    test_records = list(dict.fromkeys(args.test_records))
    overlap = sorted(set(train_records).intersection(test_records))
    if overlap:
        raise ValueError(f"Train/test records must not overlap: {', '.join(overlap)}")
    if len(train_records) + len(test_records) < 15:
        raise ValueError("Use at least 15 records for this project; provide more with --train-records/--test-records.")
    if len(train_records) + len(test_records) > 20:
        raise ValueError("Use no more than 20 records for this small project experiment.")

    records = train_records + test_records
    download_records(records)
    print("Preparing record-separated training set...")
    X_train, y_train = dataset(train_records)
    print("Preparing held-out record test set...")
    X_test, y_test = dataset(test_records)
    if len(X_train) == 0 or len(X_test) == 0:
        raise RuntimeError("No beats extracted. Verify dataset download and annotation files.")
    X_train = X_train[..., None]; X_test = X_test[..., None]
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {int(k): float(v) for k, v in zip(np.unique(y_train), weights)}
    model = build_model()
    model.summary()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(OUT_DIR / "best_model.keras"), monitor="val_loss", save_best_only=True)
    ]
    history = model.fit(X_train, y_train, validation_split=0.15, epochs=args.epochs, batch_size=args.batch_size,
                        class_weight=class_weights, callbacks=callbacks, verbose=1)
    # Test is held out by record; don't tune on these results.
    probs = model.predict(X_test, batch_size=256, verbose=1)
    pred = probs.argmax(axis=1)
    names = [LABEL_MAP[str(i)] for i in range(4)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_beats": int(len(y_test)),
        "train_beats": int(len(y_train)),
        "train_records": train_records,
        "test_records": test_records,
        "record_count": len(records),
        "dataset": "MIT-BIH Arrhythmia Database (PhysioNet)",
        "dataset_url": "https://physionet.org/content/mitdb/1.0.0/",
        "seed": 42,
        "class_names": names,
        "note": f"Educational baseline only. Record-level split; {len(records)}-record subset; not for clinical use."
    }
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "label_map.json").write_text(json.dumps(LABEL_MAP, indent=2), encoding="utf-8")
    (OUT_DIR / "classification_report.txt").write_text(classification_report(y_test, pred, labels=list(range(4)), target_names=names, zero_division=0), encoding="utf-8")
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred, labels=list(range(4)), normalize=None), display_labels=names).plot(cmap="Blues", values_format="d")
    plt.title("PulseAI — Held-out Record Confusion Matrix")
    plt.tight_layout(); plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=160); plt.close()
    plt.figure()
    plt.plot(history.history.get("loss", []), label="Training loss")
    plt.plot(history.history.get("val_loss", []), label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training history")
    plt.tight_layout(); plt.savefig(OUT_DIR / "training_history.png", dpi=160); plt.close()
    model.save(OUT_DIR / "ecg_cnn.keras")
    print("\nTRAINING FINISHED")
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts saved to: {OUT_DIR}")
    print("Copy ecg_cnn.keras and label_map.json to ../ml-service/models/.")
if __name__ == "__main__":
    main()
