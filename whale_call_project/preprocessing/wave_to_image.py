import numpy as np
import librosa
from whale_call_project.preprocessing.normalization import rms_normalize
from PIL import Image


def wave_to_spec(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Convert a waveform to a spectrogram.

    Args:
        y (np.ndarray): a 1D numpy array representing the audio waveform.
        sr (int): the sample rate of the audio waveform.

    Returns:
        np.ndarray: a 2D numpy array representing the spectrogram.
    """

    y = rms_normalize(y)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, hop_length=64, n_mels=64, fmin=100, fmax=400)

    db_mel_spec = librosa.power_to_db(mel_spec)

    return db_mel_spec


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