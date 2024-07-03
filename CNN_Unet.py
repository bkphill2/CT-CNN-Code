import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np
import os
import shutil

def pair_hilo_data(clean_dir, dirty_dir, (512,512,1):
        clean_arr = sorted(os.listdir(clean_dir))
        dirty_arr = sorted(os.listdir(dirty_dir))

        clean_path = [os.path.join(clean_dir, file) for file in clean_dir]
        dirty_path = [os.path.join(clean_dir, file) for file in dirty_dir]

        return clean_path, dirty_path

def parse_image(clean_path, dirty_path):
    clean_img = tf.io.read_file(clean_path)
    clean_img = tf.image.decode_png(clean_img, channels=3)
    clean_img = tf.image.resize(clean_img, [512, 512])
    clean_img = tf.cast(clean_img, tf.float32) / 255.0

    dirty_img = tf.io.read_file(dirty_path)
    dirty_img = tf.image.decode_png(dirty_img, channels=3)
    dirty_img = tf.image.resize(dirty_img, [512, 512])
    dirty_img = tf.cast(dirty_img, tf.float32) / 255.0

    return dirty_img, clean_img

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
cleandata_dir = '/home/NETID/bkphill2/data/hilo_paired_dataset/recons_highdose'
dirtydata_dir = '/home/NETID/bkphill2/data/hilo_paired_dataset/recons_lowdose'

clean_path, dirty_path = pair_hilo_data(clean_dir, dirty_dir)

# Directories to save the images
output_dir_original = 'output/original_images'
output_dir_reconstructed = 'output/reconstructed_images'

# Create directories if they do not exist
os.makedirs(output_dir_original, exist_ok=True)
os.makedirs(output_dir_reconstructed, exist_ok=True)


def CTUnet_model(input_shape):

        inputs = layers.Input(shape=(512, 512, 1))

        conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)

        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)

        conv3 = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(pool2)
        conv3 = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(conv3)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)

        conv4 = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(pool3)
        conv4 = layers.Conv2D(32, (2, 2), activation='relu', padding='same')(conv4)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)

        # Define the decoder
        up1 = layers.UpSampling2D((2, 2))(pool4)
        concat1 = layers.Concatenate()([up1, conv4])
        deconv1 = layers.Conv2DTranspose(32, (2, 2), activation='relu', padding='same')(concat1)
        deconv1 = layers.Conv2DTranspose(32, (2, 2), activation='relu', padding='same')(deconv1)

        up2 = layers.UpSampling2D((2, 2))(deconv1)
        concat2 = layers.Concatenate()([up2, conv3])
        deconv2 = layers.Conv2DTranspose(64, (2, 2), activation='relu', padding='same')(concat2)
        deconv2 = layers.Conv2DTranspose(64, (2, 2), activation='relu', padding='same')(deconv2)

        up3 = layers.UpSampling2D((2, 2))(deconv2)
        concat3 = layers.Concatenate()([up3, conv2])
        deconv3 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(concat3)
        deconv3 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(deconv3)

        up4 = layers.UpSampling2D((2, 2))(deconv3)
        concat4 = layers.Concatenate()([up4, conv1])
        deconv4 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(concat4)
        deconv4 = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(deconv4)

        outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(deconv4)

        Unet_Model = models.Model(inputs, outputs)

        return Unet_Model

model = CTUnet_model((512, 512, 1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(
    normalized_train_dataset,
    epochs=10,
    batch_size=16,
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

model.save('unet_model.h5')

