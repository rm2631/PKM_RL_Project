from PIL import Image
import numpy as np
import os
from datetime import datetime


def numpy_array_to_image(array):
    """
    Converts a NumPy array to an image and saves it to a folder with a filename based on the current datetime.

    Args:
        array (numpy.ndarray): The input NumPy array.
        save_folder (str): The folder where the image will be saved (default is "saved_images").
    """
    # Ensure the array data type is in the range [0, 255] (uint8)
    array = np.uint8(np.clip(array, 0, 255))

    # Create an image from the array
    image = Image.fromarray(array)
    return image
