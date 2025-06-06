import numpy as np


def peak_normalization(audio: np.ndarray) -> np.ndarray:
    """
    Normalize the audio signal to the peak amplitude
    
    Args:
    audio (np.ndarray): The audio signal to be normalized.
    
    Returns:
        np.ndarray: The normalized audio signal.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        normalized_audio = audio / peak 
        return normalized_audio
    return audio


def rms_normalize(audio: np.ndarray) -> np.ndarray:
    """
    Normalize the audio signal to the root mean square (RMS) amplitude.

    Args:
        audio (np.ndarray): The audio signal to be normalized.

    Returns:
        audio (np.ndarray): The normalized audio signal.
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        return audio / rms
    else:
        return audio


def spectrogram_normalization(spectrogram) -> np.ndarray:
    """
    Normalize the spectrogram to the range [0, 1]

    Args:
        spectrogram (np.ndarray): The spectrogram to be normalized.

    Returns:
        norm_spectrogram (np.ndarray): The normalized spectrogram.
    """

    norm_spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    return norm_spectrogram
