import librosa
import numpy as np
from preprocessing.normalization import rms_normalize


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
