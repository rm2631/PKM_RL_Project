import os
import numpy as np
from PIL import Image


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)


def _get_path(prefix, reset_id=None):
    path = os.path.join("artifacts", os.environ["run_name"], prefix)
    if reset_id:
        path = os.path.join(path, reset_id)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def flatten_channels(array):
    """
    Take an array with shape (height, width, channels) and concatenate the channels horizontally
    into a single 2D array.
    """
    if len(array.shape) == 3:
        # Split the array into a list of channels
        channels = [array[:, :, i] for i in range(array.shape[2])]
        # Concatenate the channels horizontally
        return np.concatenate(channels, axis=1)
    else:
        # If the array is already 2D, just return it
        return array


def convert_array_to_image(array):
    # Check if the array is of type float64 and convert it to uint8
    array = flatten_channels(array)
    if array.dtype == np.float64:
        # Scale the values if they are in the range 0.0 to 1.0
        if array.max() <= 1.0:
            array = (255 * array).astype(np.uint8)
        else:
            array = array.astype(np.uint8)

    # Handle 3D array with one channel
    if len(array.shape) == 3 and array.shape[2] == 1:
        # Convert to 2D array
        array = array.reshape(array.shape[0], array.shape[1])

    # Handle 2D array (grayscale)
    if len(array.shape) == 2:
        # Array is likely in the correct format for a grayscale image
        return Image.fromarray(array, "L")  # 'L' mode for grayscale images
    else:
        # Handle other cases or raise an error
        raise ValueError("Array shape is not compatible with PIL Image format")
