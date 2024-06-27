import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np


parser = argparse.ArgumentParser(description='')
parser.add_argument('--in', dest='infile', default='.', help='input file -- directory or single file')
parser.add_argument('--out', dest='outfile', default='.', help='output directory')

args = parser.parse_args()
infile, outfile = args.infile, args.outfile


if args.sup_params is None:
    use_sup = False
else:
    use_sup = True

eps = np.finfo(float).eps

if os.path.isdir(infile):               #generate list of filenames from directory
    fnames = sorted(glob(infile + '/*.flt'))
else:                                                   #single filename
    fnames = []
    fnames.append(infile)

# Define the encoder
encoder = models.Sequential([
    layers.Input(shape=(32, 32, 3)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same')
])

# Define the decoder
decoder = models.Sequential([
    layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
])

autoencoder = models.Sequential([encoder, decoder])

autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
autoencoder.fit(train_images, train_images, epochs=10, batch_size=128, shuffle=True, validation_data=(test_images, test_images))

#save image (SART code)
 #save image
    f = np.float32(f)
    f.tofile(outname)

    #save residual
    res_file.write("%f\n" % res)

    #**********save image as png**********
    max_pixel = np.amax(f)
    img = (f/max_pixel) * 255
    img = np.round(img)

    plt.figure(1)
    plt.style.use('grayscale')
    plt.imshow(img.T) #transpose image
    plt.axis('off')
    png_outname = (outname + '.png')
    plt.savefig(png_outname)
    plt.close()
