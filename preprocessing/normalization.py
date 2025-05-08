import numpy as np


def peak_normalization(audio) -> np.ndarray:
    """
    Normalize the audio signal to the peak amplitude
    
    Parameters:
    audio (numpy.ndarray): The audio signal to be normalized.
    
    Returns:
    numpy.ndarray: The normalized audio signal.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        normalized_audio = audio / peak 
        return normalized_audio
    return audio


def spectogram_normalization(spectrogram) -> np.ndarray:
    """
    Normalize the spectrogram to the range [0, 1]

    Parameters: spectrogram (numpy.ndarray): The spectrogram to be normalized.
    
    Returns:
    numpy.ndarray: The normalized spectrogram.
    """

    norm_spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    return norm_spectrogram
