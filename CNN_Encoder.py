import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import shutil

# Function to clear the contents of a directory
def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

# Path to the directory with images
data_dir = '/data/bkphill2/images'

# Directories to save the images
output_dir_original = 'output/original_images'
output_dir_reconstructed = 'output/reconstructed_images'

# Create directories if they do not exist
os.makedirs(output_dir_original, exist_ok=True)
os.makedirs(output_dir_reconstructed, exist_ok=True)

# Load the training dataset (80% of the data)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=123,
    validation_split=0.2,  # 20% of data for validation
    subset="training",     # Use this subset for training
    interpolation='bilinear'
)

# Load the validation dataset (20% of the data)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(512, 512),
    shuffle=True,
    seed=123,
    validation_split=0.2,  # 20% of data for validation
    subset="validation",   # Use this subset for validation
    interpolation='bilinear'
)

# Normalize pixel values to be between 0 and 1 using a lambda function
normalized_train_dataset = train_dataset.map(lambda x, y: (x / 255.0, x / 255.0))
normalized_validation_dataset = validation_dataset.map(lambda x, y: (x / 255.0, x / 255.0))

# Define the encoder
encoder = models.Sequential([
    layers.Input(shape=(512, 512, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2))
])

# Define the decoder
decoder = models.Sequential([
    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

# Combine encoder and decoder to create the autoencoder
autoencoder = models.Sequential([encoder, decoder])

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(
    normalized_train_dataset,
    epochs=10,
    validation_data=normalized_validation_dataset
)

# Extract a batch of images from the validation dataset
for batch in normalized_validation_dataset.take(1):
    original_images = batch[0]

# Reconstruct the images
reconstructed_images = autoencoder.predict(original_images)

# Save the original and reconstructed images
for i in range(len(original_images)):
    original_img = (original_images[i].numpy() * 255).astype(np.uint8)
    reconstructed_img = (reconstructed_images[i] * 255).astype(np.uint8)

    original_img_pil = Image.fromarray(original_img)
    reconstructed_img_pil = Image.fromarray(reconstructed_img)

    original_img_pil.save(os.path.join(output_dir_original, f'original_{i}.png'))
    reconstructed_img_pil.save(os.path.join(output_dir_reconstructed, f'reconstructed_{i}.png'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")
                                                                                 


