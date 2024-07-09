import os  #Importing needed libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from PIL import Image

def ssim_loss(y_true, y_pred): #Defining a function of ssim loss image to image
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)) #return the loss value

def mse_loss(y_true, y_pred): #Defining a mean squared error loss
    return tf.reduce_mean(tf.square(y_true - y_pred)) #Returning loss

def combined_loss(y_true, y_pred, alpha = 0.2, beta = 0.8): #Define a mixed loss with proportions alpha and beta
    return alpha * ssim_loss(y_true, y_pred) + beta * mse_loss(y_true, y_pred) #Return the sum of the weighted losses =1

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available")
else:
    print("GPU is not available")

def unet_model(input_size=(512, 512, 3)): #Defining the model
        inputs = tf.keras.Input(input_size)
            # Downsample
        ...
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs) #Initial convolutional layer
        c1 = layers.Dropout(0.1)(c1) #Drops 10% of neurons from layer 1
        c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1) #Second convolution
        p1 = layers.MaxPooling2D((2, 2))(c1) #Max pools 2x2 regions

        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1) #Second Same as first section but filters x 2
        c2 = layers.Dropout(0.1)(c2)
        c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2) #Third same but filters x 2
        c3 = layers.Dropout(0.1)(c3)
        c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3) #Fourth same but filters x 2
        c4 = layers.Dropout(0.1)(c4)
        c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4) #Fourth has no maxpool and filters x 2
        c5 = layers.Dropout(0.1)(c5)
        c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5) #Undoes previous convolve
        u6 = layers.concatenate([u6, c4], axis=3) #First skip connection
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6) #Filtering convolutional layer
        c6 = layers.Dropout(0.1)(c6) #Drop 10% neurons from layer 6
        c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6) #Second convolution and filter

        u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6) #Same as first but half filters
        u7 = layers.concatenate([u7, c3], axis=3) #Second skip connection
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = layers.Dropout(0.1)(c7)
        c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7) #Same but half filters
        u8 = layers.concatenate([u8, c2], axis=3)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = layers.Dropout(0.1)(c8)
        c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8) #Same but half filters
        u9 = layers.concatenate([u9, c1], axis=3)
        c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = layers.Dropout(0.1)(c9)
        c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)


        outputs = layers.Conv2D(3, (1, 1), activation='sigmoid')(c9) #Final 3 filter 1x1 kernel with sigmoid activation
        model = models.Model(inputs=inputs, outputs=[outputs]) #Compresses the function into a single variable "model"

        return model #Returns that variable

def load_images_from_directory(directory, target_size=(512, 512)): #Defines a data loading function with parameters "directory" and size of image
    images = []                   #Creates an empty array called images
    for filename in os.listdir(directory): #For each file in the directory argument do
        if filename.endswith(".png"): #If it is a png file (This can be modified to flt)
            img = load_img(os.path.join(directory, filename), target_size=target_size) #Loads the images from the directory given
            img = img_to_array(img)       #defines the img variable as the argument of an image to array function
            img = img / 255.0  #Normalizes pixel values
            images.append(img)  #adds each file in the directory to the end of the array in order
    return np.array(images) #returns the array of images

# Directories
clean_dir = '/mmfs1/gscratch/uwb/bkphill2/recons_highdose' #Local directory of high dose images
dirty_dir = '/mmfs1/gscratch/uwb/bkphill2/recons_lowdose' #Local directory of low dose images

clean_images = load_images_from_directory(clean_dir) #Applies the load image function to the clean images and names the array
dirty_images = load_images_from_directory(dirty_dir) #Does the same for the dirty

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(dirty_images, clean_images, test_size=0.2, random_state=42)

#defines model as the unet model
model = unet_model()
model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy']) #Compiles the model with adam optimizer and our mixed loss

# Train model
model.fit(X_train, y_train, validation_split=0.1, epochs=750, batch_size=16) #Trains the dirty images with clean images as target

# Save model
model.save('unet_model.h5') #Saves the model

# Creates images run through the model
predicted_images = model.predict(X_test)

# Save original and reconstructed images for comparison
output_dir_original = '/mmfs1/home/bkphill2/output/original_images' #A directory to store the original training data
output_dir_reconstructed = '/mmfs1/home/bkphill2/output/reconstructed_images' #A directory to store the reconstructed data
for i in range(len(X_test)): #For all indices in the range of the test array
    original_img = (X_test[i] * 255).astype(np.uint8) #Takes images back into standard pixel values. Saves as int
    reconstructed_img = (predicted_images[i] * 255).astype(np.uint8) #Takes predicted images into standard pixel values and saves as int

    original_img_pil = Image.fromarray(original_img)   #Takes the images out of the array
    reconstructed_img_pil = Image.fromarray(reconstructed_img)
    original_img_pil.save(os.path.join(output_dir_original, f'original_{i}.png')) #Saves these images as pngs of index i and places them into directorys
    reconstructed_img_pil.save(os.path.join(output_dir_reconstructed, f'reconstructed_{i}.png'))

print(f"Saved original images to {output_dir_original}") #Prints if save succesful
print(f"Saved reconstructed images to {output_dir_reconstructed}")
