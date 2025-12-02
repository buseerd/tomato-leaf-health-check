import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATA_DIR = Path(r"C:\src\plant_disease_project\data")

healthy_dir = DATA_DIR / "healthy"
early_dir   = DATA_DIR / "early_blight"

print("Healthy klasörü :", healthy_dir)
print("Early_blight klasörü:", early_dir)

def list_images(folder: Path):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return files

healthy_files = list_images(healthy_dir)
early_files   = list_images(early_dir)

print("Healthy görüntü sayısı    :", len(healthy_files))
print("Early_blight görüntü sayısı:", len(early_files))


filepaths = healthy_files + early_files
labels    = [0] * len(healthy_files) + [1] * len(early_files)

df = pd.DataFrame({
    "filepath": [str(p) for p in filepaths],
    "label": labels
})

print("\nİlk 5 kayıt:")
print(df.head())
print("\nSınıf dağılımı:")
print(df["label"].value_counts())



#----------------
# 0 -> "healthy", 1 -> "early_blight"
df["label_str"] = df["label"].map({0: "healthy", 1: "early_blight"})

print("\nString label dağılımı:")
print(df["label_str"].value_counts())

# -------------------------------------------------------------------
# 4) GÖRÜNTÜ BOYUTU VE BATCH SIZE
# -------------------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH  = 224
BATCH_SIZE = 32

# -------------------------------------------------------------------
# TRAIN / TEST AYRIMI (şimdilik k-fold yok)
# -------------------------------------------------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

print("\nTrain örnek sayısı:", len(train_df))
print("Test örnek sayısı :", len(test_df))

# -------------------------------------------------------------------
# -------------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(
    rescale=1.0/255
)

# -------------------------------------------------------------------
# -------------------------------------------------------------------
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="filepath",
    y_col="label_str",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col="filepath",
    y_col="label_str",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------------------------------------------------------
# -------------------------------------------------------------------
images, labels_batch = next(train_generator)

print("\nBir batch görüntü şekli:", images.shape)
print("İlk 10 label:", labels_batch[:10])

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(f"Label: {int(labels_batch[i])}")  # 0=healthy, 1=early_blight
    plt.axis("off")
plt.tight_layout()
plt.show()
