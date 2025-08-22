# import os
# from PIL import Image
#
# # Folosim folderele tale
# input_folder = "/home/vladut/Desktop/pandas/PetImages/Dog"
# output_folder = "/home/vladut/Desktop/pandas/dogs"
#
# os.makedirs(output_folder, exist_ok=True)
#
# processed = 0
# skipped = 0
#
# for filename in os.listdir(input_folder):
#     if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         img_path = os.path.join(input_folder, filename)
#         try:
#             img = Image.open(img_path)
#             img = img.convert("RGB")  # Conversie pentru siguranță
#             img_resized = img.resize((128, 128))
#
#             save_path = os.path.join(output_folder, filename)
#             img_resized.save(save_path)
#             processed += 1
#         except Exception as e:
#             print(f"⚠️ Fișier sărit: {filename} (eroare: {e})")
#             skipped += 1
#
# print(f"✅ Procesarea s-a terminat. Imagini salvate în: {output_folder}")
# print(f"   - Procesate cu succes: {processed}")
# print(f"   - Sărite (corupte/eronate): {skipped}")

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Folderele tale
train_dir = "train"   # are subfoldere: cats/ și dogs/
test_dir = "test"     # imagini amestecate

# Parametrii
img_size = (128, 128)
batch_size = 32

# Încarcăm datele de antrenament
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="binary"
)

# Normalizare (0-1)
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# Model simplu CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # 0 = pisică, 1 = câine
])

# Compilăm
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Antrenare
history = model.fit(
    train_ds,
    epochs=10
)

# Numele claselor
class_names = ["cat", "dog"]

# Predicții pe imaginile din test/
for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)

    try:
        # Încărcăm imaginea și o redimensionăm
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalizare [0,1]

        # Facem predicția
        prediction = model.predict(img_array, verbose=0)[0][0]
        label_pred = class_names[int(prediction > 0.5)]

        print(f"📷 {filename} → Predicție: {label_pred}")
    except Exception as e:
        print(f"⚠️ Nu am putut procesa {filename}: {e}")
