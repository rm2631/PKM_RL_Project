from PIL import Image
import numpy as np
import os
from datetime import datetime


def numpy_array_to_image_and_save(array, save_folder="saved_images"):
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

    # Generate a filename based on the current datetime
    current_datetime = datetime.now()
    filename = current_datetime.strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"

    # Create the save path by joining the folder and filename
    save_path = os.path.join(save_folder, filename)

    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the image to the specified folder with the generated filename
    image.save(save_path)
