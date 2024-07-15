import os #Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from PIL import Image

# Initialize the MirroredStrategy, this theoretically forces the gpus to run in parallel
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#gpu debug check
gpus = tf.config.experimental.list_physical_devices('GPU') #gpus is boolean set true if 'gpu' is seen
if gpus: #if true
    try:
        # Enable all visible GPUs
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        for gpu in gpus:  #For all GPUs
            tf.config.experimental.set_memory_growth(gpu, True) #Set memory growth per gpu
        print("Utilizing all visible GPUs")
    except RuntimeError as e: #If failure, prints error
        print(e)
else:   #Prints if no gpu
    print("No GPUs available")

# Check if GPU is available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #Uses all gpus by indexing to the back

with strategy.scope(): #Uses the parallelization strat
    def create_patch_embeddings(inputs, patch_size=2, projection_dim=64): #Defines patch embedding function
        # Assuming inputs have shape (batch_size, height, width, channels)
        batch_size, height, width, channels = inputs.shape
        # Calculate the number of patches
        num_patches = (height // patch_size) * (width // patch_size)
        patches = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size)(inputs) #Convolves a filter that generates patches of the image
        # Reshape the output to have (num_patches, projection_dim)
        patches = layers.Reshape((num_patches, projection_dim))(patches)
              return patches

    def positional_encoding(num_patches, projection_dim): #Def function for positional encoding the patches generated
        position = np.arange(num_patches)[:, np.newaxis] #Creates an array of the range up to num_patches, then makes it a 2d column vector
        div_term = np.exp(np.arange(0, projection_dim, 2) * -(np.log(10000.0) / projection_dim)) #This is the coeffient in the arg of the trig funcs based on pos
        pos_enc = np.zeros((num_patches, projection_dim)) #Creates an array of zeroes the same size as patches array
        pos_enc[:, 0::2] = np.sin(position * div_term) #Even indices are unique sine frequencies
        pos_enc[:, 1::2] = np.cos(position * div_term) #Odd indices are unique cos frequencies
        pos_enc = pos_enc[np.newaxis, ...]  # Shape: [1, num_patches, projection_dim]
        return tf.cast(pos_enc, dtype=tf.float32) #Casts as float for use in transformer

    class TransformerBlock(layers.Layer):  #Creates the transformer block class
        def __init__(self, projection_dim, num_heads): #constructor  method with params of self, the dim of embedding space, and number of transformer heads
            super(TransformerBlock, self).__init__() #calls layers.layer parent class and init.
            self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim) #Attention layer
            self.ffn = tf.keras.Sequential([ #Dense forward projection/multilayer perceptron layer
                layers.Dense(projection_dim, activation="relu"),
                layers.Dense(projection_dim),
            ])
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) #Normalizes perceptron outputs
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        def call(self, inputs): #Defines method to add attention values to initial input vectors
            attn_output = self.attention(inputs, inputs) #creates a variable of the attention output
            out1 = self.layernorm1(inputs + attn_output) #adds to original and normalizes and stores as out1
            ffn_output = self.ffn(out1) #Runs this through feed forward
            return self.layernorm2(out1 + ffn_output) #Adds that output to output of attention layer

    def build_transformer_model(num_patches, projection_dim, num_heads, transformer_layers, num_classes): #Function to build the transformer model

        inputs = layers.Input(shape=(512, 512, 1))  #Defines image shape as 512x512x1 tensor
        patches = create_patch_embeddings(inputs, patch_size=8, projection_dim=projection_dim) #Defines patches as patch embedding function called on images
        encoded_patches = patches + positional_encoding(num_patches, projection_dim)

        x = encoded_patches  #Variable storage of encoded patches
        for _ in range(transformer_layers): #For ever transformer layer
            x = TransformerBlock(projection_dim, num_heads)(x) #Iterate the transformer block on the patches

        # Use Conv2D and UpSampling2D layers to reconstruct the image
        x = layers.Reshape((64, 64, 64))(x) #Reshape as 64x64x64 tensor

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x) #Filter that image with 128 3x3 filters
        x = layers.UpSampling2D((2, 2))(x)  #Upsamples image to 128x128
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x) #Second filter pass
        x = layers.UpSampling2D((4, 4))(x)  #Gets us back to 512x512 image
        outputs = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output layer to reconstruct the image

        return tf.keras.Model(inputs, outputs)  # Return the model object

    def load_flt_file(file_path, shape=(512, 512), add_channel=True): #Defines the flt load function with image size
        with open(file_path, 'rb') as file: #Opens the directory containing the flts
            data = np.fromfile(file, dtype=np.float32) #Stores the files as floats
            img = data.reshape(shape) #Reshapes the data to be used
            if add_channel: #Add channel set true so we do
                img = img[:, :, np.newaxis] #Adds a channel of depth to flt for use
        return img

    # Load and preprocess images
    def load_images_from_directory(directory, target_size=(512, 512)): #Function to load the images
        images = []                                                             #Creates empty array to store images
        for filename in os.listdir(directory): #For each file in the directory do
            if filename.endswith(".flt"): #If the file is an flt
                file_path = os.path.join(directory, filename) #Stores the name of the directory and file path to load
                img = load_flt_file(file_path, shape=target_size) #Loads the flts using previous function and stores as 512x512
                img = img / 255.0 #Normalizes to values between [0,1]
                images.append(img) #Adds the grabbed image to back of array
        return np.array(images)

    # Directories
    clean_dir = '/mmfs1/gscratch/uwb/bkphill2/900_views_flt' #Sets the data directories
    dirty_dir = '/mmfs1/gscratch/uwb/bkphill2/60_views_flt'

    clean_images = load_images_from_directory(clean_dir) #Performs the load image function on data directories
    dirty_images = load_images_from_directory(dirty_dir)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(dirty_images, clean_images, test_size=0.2, random_state=42) #Sets the training values and seed

    # Build and compile the transformer model
    num_patches = 4096 #Defines the number of patches used
    projection_dim = 64 #This is the dimension of the embedding space of the patches (Where the patch vectors exist)
    num_heads = 4 #Number of heads in the transformer
    transformer_layers = 6 #How many transformer layers
    num_classes = 4 #Not sure if this line is useful

    model = build_transformer_model(num_patches, projection_dim, num_heads, transformer_layers, num_classes) #Builds the model with our attributes
    model.compile(optimizer='adam', loss='mean_squared_error') #Uses mse loss


    # Train model
    model.fit(X_train, y_train, validation_split=0.1, epochs=250, batch_size=12)

    # Save model
    model.save('transformer_model.h5')

# Predict on a test set
predicted_images = model.predict(X_test)

# Save original and reconstructed images for comparison
output_dir_original = '/mmfs1/home/bkphill2/output/original_images' #Saves the originals and reconstructed images to directory (Commented in other git code)
output_dir_reconstructed = '/mmfs1/home/bkphill2/output/reconstructed_images'

def save_as_flt(data, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Open the file in write-binary (wb) mode
    with open(file_path, 'wb') as file:
    # Write the data as floating point data
        file.write(data.astype(np.float32).tobytes())

for i in range(len(X_test)):
    original_img = X_test[i] * 255  # Scale if necessary, but keep as float if saving as .flt
    reconstructed_img = predicted_images[i] * 255  # Scale if necessary, but keep as float

    save_as_flt(original_img, os.path.join(output_dir_original, f'original_{i}.flt'))
    save_as_flt(reconstructed_img, os.path.join(output_dir_reconstructed, f'reconstructed_{i}.flt'))

print(f"Saved original images to {output_dir_original}")
print(f"Saved reconstructed images to {output_dir_reconstructed}")
