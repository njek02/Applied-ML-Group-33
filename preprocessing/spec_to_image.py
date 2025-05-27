import numpy as np
from PIL import Image


def spec_to_image(spectrogram: np.ndarray) -> np.ndarray:
    """
    Convert a spectrogram to an image.

    Args:
        spectrogram (np.ndarray): The input spectrogram.

    Returns:
        input_image (np.ndarray): The output image representation of the spectrogram.
    """    
    # Normalize spectrogram to [0, 255]
    spec_image = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) * 255
    spec_image = spec_image.astype(np.uint8)

    # Convert to Image class and resize image
    image = Image.fromarray(spec_image)
    resized_image = image.resize((32, 32))

    # Convert back to array and normalize to [0, 1]
    input_image = np.array(resized_image).astype(np.float32) / 255.0

    # Reshape to fit a PyTorch CNN
    input_image = input_image.reshape((1, 32, 32))

    return input_image
