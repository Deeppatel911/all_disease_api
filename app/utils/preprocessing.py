from skimage.io import imread
from skimage.transform import resize
from io import BytesIO
import numpy as np

MEAN = np.array([0.824523, 0.854478, 0.938681])
STD = np.array([0.148047, 0.152414, 0.057466])
IMG_SIZE = (128, 128)

def preprocess_image(image_bytes):
    # Convert bytes → file-like object
    img = imread(BytesIO(image_bytes))

    # Resize
    img_resized = resize(img, IMG_SIZE, preserve_range=True)

    # Normalize
    img_norm = (img_resized - MEAN) / (STD + 1e-7)

    # Add batch dimension
    img_final = img_norm.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)

    return img_final
