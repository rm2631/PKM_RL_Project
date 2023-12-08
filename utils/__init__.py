import os
import numpy as np
from PIL import Image
from datetime import datetime
import json

# from stable_baselines3.common.monitor import Monitor
# from utils.PkmEnv2 import PkmEnv2


def print_section(text):
    print("" * 80)
    print("=" * 80)
    print(text)
    print("=" * 80)
    print("" * 80)


def set_run_name(run_name=None):
    if not run_name:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    os.environ["run_name"] = run_name
    return run_name


def ndarray_serializer(obj):
    """Custom serializer for numpy objects"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def save_log(file_path, data_list):
    """
    Saves a list of dictionaries to a JSON file.
    This function can handle numpy objects inside the dictionaries.

    :param file_path: Path to the JSON file to be saved.
    :param data_list: List of dictionaries to be saved.
    """
    try:
        with open(file_path, "w") as file:
            json.dump(data_list, file, default=ndarray_serializer)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


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
