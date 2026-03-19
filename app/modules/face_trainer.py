import threading
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import app.config as cfg

DATA_PATH    = Path(__file__).parent.parent.parent / "data" / "gestures.csv"
MODEL_DIR    = Path(__file__).parent.parent.parent / "data" / "models"

FACE_START = 253   # first face landmark column index (after pose+both_hands)
FACE_END   = 1657  # one past last face landmark column index (1404 dims total)


class FaceTrainingSession:
    def __init__(self, epochs: int = 50, data_path: Path = None, model_dir: Path = None):
        self.epochs         = epochs
        self._data_path     = data_path or DATA_PATH
        self._model_dir     = model_dir or MODEL_DIR
        self._model_path    = self._model_dir / "face_model.h5"
        self._encoder_path  = self._model_dir / "face_label_encoder.pkl"
        self.status         = "idle"
        self.progress       = 0.0
        self.logs           = []
        self.history        = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
        self.final_accuracy = None
        self.message        = ""

    def start(self):
        self.status = "running"
        threading.Thread(target=self._train, daemon=True).start()

    def _train(self):
        try:
            from tensorflow.keras.utils import to_categorical
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import Callback

            self.logs.append("Loading data...")
            df = pd.read_csv(self._data_path)
            if len(df) < 10:
                raise ValueError("Not enough training data. Record more gestures first.")

            df["unique_id"] = df["class"] + "_" + df["sequence_id"].astype(str)
            unique_ids = df["unique_id"].unique()

            sequences, labels = [], []
            for uid in unique_ids:
                seq_df   = df[df["unique_id"] == uid]
                all_feat = seq_df.drop(columns=["class", "sequence_id", "unique_id"]).values
                # Slice only face landmark columns
                sequences.append(all_feat[:, FACE_START:FACE_END])
                labels.append(seq_df["class"].iloc[0])

            X = pad_sequences(sequences, padding="post", dtype="float32", value=-1.0)
            y = np.array(labels)

            le    = LabelEncoder()
            y_enc = le.fit_transform(y)
            y_oh  = to_categorical(y_enc)

            self.logs.append(f"Classes: {list(le.classes_)}")
            self.logs.append(f"Total sequences: {len(sequences)}")

            if len(np.unique(y_enc)) < 2:
                raise ValueError("Need at least 2 gesture classes to train.")

            class_counts = np.bincount(y_enc)
            min_count    = int(class_counts.min())
            if min_count < 2:
                sparse = [le.classes_[i] for i, c in enumerate(class_counts) if c < 2]
                raise ValueError(
                    f"Each gesture needs at least 2 recordings before training. "
                    f"These have only 1: {sparse}. Record more sessions and try again."
                )

            n_test        = max(1, int(len(sequences) * 0.2))
            use_stratify  = min_count >= 2 and n_test >= len(le.classes_)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_oh, test_size=0.2, random_state=42,
                stratify=y_enc if use_stratify else None
            )

            cw = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)

            model = Sequential([
                Masking(mask_value=-1.0, input_shape=(X_train.shape[1], X_train.shape[2])),
                LSTM(64, return_sequences=True),
                Dropout(0.3),
                LSTM(64),
                Dropout(0.3),
                Dense(y_train.shape[1], activation="softmax"),
            ])
            model.compile(
                optimizer=Adam(learning_rate=cfg.LEARNING_RATE),
                loss="categorical_crossentropy", metrics=["accuracy"]
            )

            session = self

            class _CB(Callback):
                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    session.progress = (epoch + 1) / session.epochs * 100
                    session.history["accuracy"].append(float(logs.get("accuracy", 0)))
                    session.history["val_accuracy"].append(float(logs.get("val_accuracy", 0)))
                    session.history["loss"].append(float(logs.get("loss", 0)))
                    session.history["val_loss"].append(float(logs.get("val_loss", 0)))
                    session.logs.append(
                        f"Epoch {epoch+1}/{session.epochs} — "
                        f"acc: {logs.get('accuracy',0):.4f}, "
                        f"val_acc: {logs.get('val_accuracy',0):.4f}"
                    )

            self.logs.append("Training started...")
            model.fit(
                X_train, y_train,
                epochs=self.epochs, batch_size=cfg.BATCH_SIZE,
                validation_data=(X_test, y_test),
                class_weight=dict(enumerate(cw)),
                callbacks=[_CB()], verbose=0
            )

            self._model_dir.mkdir(parents=True, exist_ok=True)
            model.save(self._model_path)
            joblib.dump(le, self._encoder_path)

            _, acc = model.evaluate(X_test, y_test, verbose=0)
            self.final_accuracy = float(acc)
            self.status  = "done"
            self.message = f"Training complete! Test accuracy: {acc*100:.1f}%"
            self.logs.append(self.message)

        except Exception as e:
            self.status  = "error"
            self.message = str(e)
            self.logs.append(f"Error: {e}")


current_training: FaceTrainingSession = None
