import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# 1. Load the Processed Dataset
# We use color_mode='grayscale' because our DIP pipeline produced gray images
data_dir = 'processed_dataset'
img_size = (64, 64)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

# 2. Build the CNN Architecture
# This is a 3-layer network perfect for medical image classification
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(64, 64, 1)), # Normalize pixels
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5), # Prevents overfitting (Member 3's task!)
    layers.Dense(1, activation='sigmoid') # Binary output: 0 (Normal) or 1 (Anomaly)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 3. Train the Model
epochs = 15
print("\nStarting Training Stage...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 4. Save Results for your Report
model.save('heartbeat_model.h5')
print("\nModel saved as heartbeat_model.h5")

# Plot Accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Final Model Accuracy')
plt.legend()
plt.savefig('accuracy_graph.png')
print("Accuracy graph saved as accuracy_graph.png")