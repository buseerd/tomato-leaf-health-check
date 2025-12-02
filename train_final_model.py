from pathlib import Path

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# -------------------------------
# 1) GENEL AYARLAR
# -------------------------------
DATA_DIR = Path(r"C:\src\plant_disease_project\data")

IMG_HEIGHT = 128
IMG_WIDTH  = 128
BATCH_SIZE = 16
EPOCHS     = 5  

# -------------------------------
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2   # %80 train, %20 validation
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
# 5) MODELİ KAYDET
# -------------------------------
model.save("leaf_model.h5")
print("Model kaydedildi: leaf_model.h5")
