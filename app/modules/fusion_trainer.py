import threading
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "gestures.csv"
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"

FACE_START = 176
FACE_END   = 1580


class FusionTrainingSession:
    """
    Trains a small Dense meta-learner on top of the three LSTM softmax outputs.

    Input:  concat([softmax_mv, softmax_face, softmax_au])  shape: (3 * n_classes,)
    Output: Dense(32, relu) → Dropout(0.3) → Dense(n_classes, softmax)

    Training data is built from gestures.csv (movement + face streams).
    Audio softmax is filled with a uniform prior (1/n_classes) because gestures.csv
    has no paired audio. At inference time the real audio softmax is passed in.
    If an audio model exists with matching classes, its softmax is used instead of
    the uniform prior for sequences taken from audio_gestures.csv, which are also
    added to the training set.
    """

    def __init__(self, epochs: int = 50, data_path: Path = None, model_dir: Path = None):
        self.epochs         = epochs
        self._data_path     = data_path or DATA_PATH
        self._model_dir     = model_dir or MODEL_DIR
        self._model_path    = self._model_dir / "fusion_model.h5"
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
            from tensorflow.keras.models import Sequential, load_model
            from tensorflow.keras.layers import Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import Callback

            # ── Load base models ──────────────────────────────────────────────
            mv_model_path = self._model_dir / "movement_model.h5"
            mv_enc_path   = self._model_dir / "label_encoder.pkl"
            fa_model_path = self._model_dir / "face_model.h5"
            fa_enc_path   = self._model_dir / "face_label_encoder.pkl"
            au_model_path = self._model_dir / "audio_model.h5"
            au_enc_path   = self._model_dir / "audio_label_encoder.pkl"

            if not mv_model_path.exists():
                raise FileNotFoundError("Movement model not found. Train the movement model first.")
            if not fa_model_path.exists():
                raise FileNotFoundError("Face model not found. Train the face model first.")

            self.logs.append("Loading movement model...")
            mv_model = load_model(mv_model_path)
            mv_le    = joblib.load(mv_enc_path)
            n_classes = len(mv_le.classes_)

            self.logs.append("Loading face model...")
            fa_model = load_model(fa_model_path)
            fa_le    = joblib.load(fa_enc_path)

            if list(fa_le.classes_) != list(mv_le.classes_):
                raise ValueError(
                    "Face model and movement model have different gesture classes. "
                    "Retrain both models on the same gesture set."
                )

            au_model = None
            au_le    = None
            if au_model_path.exists():
                self.logs.append("Loading audio model...")
                au_model = load_model(au_model_path)
                au_le    = joblib.load(au_enc_path)
                if list(au_le.classes_) != list(mv_le.classes_):
                    self.logs.append(
                        "Warning: audio model classes differ from movement classes — "
                        "using uniform prior for audio stream."
                    )
                    au_model = None  # treat as missing

            # ── Build training set from gestures.csv ──────────────────────────
            self.logs.append("Loading gesture data...")
            df = pd.read_csv(self._data_path)
            if len(df) < 10:
                raise ValueError("Not enough training data.")

            df["unique_id"] = df["class"] + "_" + df["sequence_id"].astype(str)
            unique_ids      = df["unique_id"].unique()

            X_meta, y_meta = [], []
            uniform_au = np.full(n_classes, 1.0 / n_classes, dtype="float32")

            # Expected sequence lengths from trained models
            mv_seq_len = mv_model.input_shape[1]
            fa_seq_len = fa_model.input_shape[1]

            for uid in unique_ids:
                seq_df   = df[df["unique_id"] == uid]
                all_feat = seq_df.drop(columns=["class", "sequence_id", "unique_id"]).values
                label    = seq_df["class"].iloc[0]

                if label not in mv_le.classes_:
                    continue

                # Movement softmax
                mv_seq = pad_sequences(
                    [all_feat], maxlen=mv_seq_len, padding="post",
                    dtype="float32", value=-1.0
                )
                softmax_mv = mv_model.predict(mv_seq, verbose=0)[0]

                # Face softmax
                face_feat = all_feat[:, FACE_START:FACE_END]
                fa_seq    = pad_sequences(
                    [face_feat], maxlen=fa_seq_len, padding="post",
                    dtype="float32", value=-1.0
                )
                softmax_fa = fa_model.predict(fa_seq, verbose=0)[0]

                # Audio softmax — uniform prior (gestures.csv has no audio)
                softmax_au = uniform_au

                combined = np.concatenate([softmax_mv, softmax_fa, softmax_au])
                X_meta.append(combined)
                y_meta.append(label)

            # ── Optionally augment with audio_gestures.csv sequences ──────────
            audio_csv = self._data_path.parent / "audio_gestures.csv"
            if au_model is not None and audio_csv.exists():
                try:
                    self.logs.append("Adding audio gesture sequences to training set...")
                    adf = pd.read_csv(audio_csv)
                    adf["unique_id"] = adf["class"] + "_" + adf["sequence_id"].astype(str)
                    au_seq_len = au_model.input_shape[1]

                    uniform_mv_fa = np.full(n_classes, 1.0 / n_classes, dtype="float32")

                    for uid in adf["unique_id"].unique():
                        sub   = adf[adf["unique_id"] == uid]
                        label = sub["class"].iloc[0]
                        if label not in au_le.classes_:
                            continue
                        au_feat = sub.drop(columns=["class", "sequence_id", "unique_id"]).values
                        au_seq  = pad_sequences(
                            [au_feat], maxlen=au_seq_len, padding="post",
                            dtype="float32", value=0.0
                        )
                        softmax_au = au_model.predict(au_seq, verbose=0)[0]
                        combined   = np.concatenate([uniform_mv_fa, uniform_mv_fa, softmax_au])
                        X_meta.append(combined)
                        y_meta.append(label)
                except Exception as e:
                    self.logs.append(f"Could not load audio gestures: {e}")

            if len(X_meta) < 4:
                raise ValueError("Not enough sequences to train the fusion meta-learner.")

            X = np.array(X_meta, dtype="float32")
            y = np.array(y_meta)

            from sklearn.preprocessing import LabelEncoder
            le    = LabelEncoder()
            le.classes_ = mv_le.classes_   # use movement model's ordering
            y_enc = le.transform(y)
            y_oh  = to_categorical(y_enc, num_classes=n_classes)

            self.logs.append(f"Meta-learner training samples: {len(X)}")
            self.logs.append(f"Classes: {list(le.classes_)}")

            if len(np.unique(y_enc)) < 2:
                raise ValueError("Need at least 2 gesture classes.")

            class_counts = np.bincount(y_enc)
            min_count    = int(class_counts.min())
            n_test        = max(1, int(len(X) * 0.2))
            use_stratify  = min_count >= 2 and n_test >= n_classes
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_oh, test_size=0.2, random_state=42,
                stratify=y_enc if use_stratify else None
            )

            cw = compute_class_weight("balanced", classes=np.unique(y_enc), y=y_enc)

            meta = Sequential([
                Dense(32, activation="relu", input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(n_classes, activation="softmax"),
            ])
            meta.compile(
                optimizer=Adam(learning_rate=0.001),
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

            self.logs.append("Training meta-learner...")
            meta.fit(
                X_train, y_train,
                epochs=self.epochs, batch_size=32,
                validation_data=(X_test, y_test),
                class_weight=dict(enumerate(cw)),
                callbacks=[_CB()], verbose=0
            )

            self._model_dir.mkdir(parents=True, exist_ok=True)
            meta.save(self._model_path)

            _, acc = meta.evaluate(X_test, y_test, verbose=0)
            self.final_accuracy = float(acc)
            self.status  = "done"
            self.message = f"Training complete! Test accuracy: {acc*100:.1f}%"
            self.logs.append(self.message)

        except Exception as e:
            self.status  = "error"
            self.message = str(e)
            self.logs.append(f"Error: {e}")


current_training: FusionTrainingSession = None
