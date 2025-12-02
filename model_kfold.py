from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# --------------------------------------------------------
# --------------------------------------------------------
DATA_DIR = Path(r"C:\src\plant_disease_project\data")

healthy_dir = DATA_DIR / "healthy"
early_dir   = DATA_DIR / "early_blight"

def list_images(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return files

healthy_files = list_images(healthy_dir)
early_files   = list_images(early_dir)

filepaths = healthy_files + early_files
labels    = [0] * len(healthy_files) + [1] * len(early_files)   # 0=healthy, 1=early_blight

df = pd.DataFrame({
    "filepath": [str(p) for p in filepaths],
    "label": labels
})

df["label_str"] = df["label"].map({0: "healthy", 1: "early_blight"})


df_small = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=min(len(x), 1200), random_state=42))
      .reset_index(drop=True)
)

df = df_small

print("\nKısıtlanmış veri seti boyutu:", len(df))
print(df["label"].value_counts())



print("Toplam örnek sayısı:", len(df))
print("Sınıf dağılımı:")
print(df["label"].value_counts())

# --------------------------------------------------------
# --------------------------------------------------------
IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 16
EPOCHS     = 3

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255
)

# --------------------------------------------------------
# --------------------------------------------------------
def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")  # binary çıktı
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# --------------------------------------------------------
# --------------------------------------------------------
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold_no = 1
f1_scores = []
acc_scores = []

X = df["filepath"].values
y = df["label"].values

for train_index, val_index in skf.split(X, y):
    print("\n" + "="*40)
    print(f"FOLD {fold_no}")
    print("="*40)

    train_df = df.iloc[train_index].reset_index(drop=True)
    val_df   = df.iloc[val_index].reset_index(drop=True)

    # Generator'lar
    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="label_str",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="label_str",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode="binary",
        batch_size=BATCH_SIZE,
        shuffle=False      # ÖNEMLİ: tahminlerle sırayı eşleştirmek için
    )

    # Modeli her fold için sıfırdan kur
    model = build_model()

    # Eğitim
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        verbose=1
    )

    # Validation set üzerinde tahmin al
    y_prob = model.predict(val_gen)
    y_pred = (y_prob > 0.5).astype(int).ravel()   # 0/1'e çevir

    y_true = val_df["label"].values

    # F1 skoru (pozitif sınıf: 1 = early_blight)
    f1 = f1_score(y_true, y_pred, average="binary")
    loss, acc = model.evaluate(val_gen, verbose=0)

    print(f"Fold {fold_no} - Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    f1_scores.append(f1)
    acc_scores.append(acc)

    fold_no += 1

# --------------------------------------------------------
# TÜM FOLD'LARIN ORTALAMA SONUÇLARI
# --------------------------------------------------------
print("\n" + "#"*50)
print("K-FOLD SONUÇLARI")
print("#"*50)
print(f"F1 skorları (her fold): {['{:.4f}'.format(s) for s in f1_scores]}")
print(f"Ortalama F1 skoru      : {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Doğruluk (Accuracy)    : {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
