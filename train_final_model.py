from pathlib import Path
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


# -------------------------------
# 1) GENEL AYARLAR
# -------------------------------
DATA_DIR = Path(r"C:\src\plant_disease_project\data")

IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 16
EPOCHS     = 5


# -------------------------------
# 2) DATA GENERATOR (Ön işleme + Augmentation)
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.15,
    shear_range=0.08,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    channel_shift_range=20.0,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Sınıf indeksleri:", train_gen.class_indices)
# Örn: {'early_blight': 0, 'healthy': 1}


# -------------------------------
# 3) CNN MODELİ
# -------------------------------
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
    layers.Dense(1, activation="sigmoid")   # binary çıktı
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -------------------------------
# 4) EĞİTİM
# -------------------------------
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=1
)


# -------------------------------
# 5) PERFORMANS METRİKLERİ (Validation)
#    Confusion Matrix + Precision/Recall/F1 + Accuracy
# -------------------------------
val_gen.reset()

y_true = val_gen.classes  # gerçek etiketler (0/1)
y_prob = model.predict(val_gen, verbose=0).ravel()  # 0-1 arası olasılık
y_pred = (y_prob >= 0.5).astype(int)  # 0/1 tahmin

print("\n==============================")
print("PERFORMANS METRİKLERİ (Validation)")
print("==============================")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# class_indices: {'early_blight':0, 'healthy':1} gibi
# report'a isimleri doğru vermek için label sırasını garantiye alıyoruz
idx_to_class = {v: k for k, v in val_gen.class_indices.items()}
target_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1-score:", f1_score(y_true, y_pred))


# -------------------------------
# 6) MODELİ KAYDET
# -------------------------------
model.save("leaf_model.h5")
print("\nModel kaydedildi: leaf_model.h5")
