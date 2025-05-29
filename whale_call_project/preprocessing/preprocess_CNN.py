import librosa
import numpy as np
from whale_call_project.preprocessing.wave_to_image import wave_to_spec, spec_to_image


def preprocess_sample(file_path: str) -> np.ndarray:
    """Preprocess audio file for CNN input.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        spec_image (np.ndarray): Preprocessed audio file as a spectrogram image.
    """    
    y, sr = librosa.load(file_path, sr=None, duration=2.0)

    if sr != 2000:
        raise ValueError(f"Invalid sampling rate: {sr}. Expected 2000 Hz.")

    if len(y) < 4000:
        raise ValueError(f"Audio fragment too short: {len(y)/sr:.4f}. Minimum duration needs to be 2 seconds")

    spectrogram = wave_to_spec(y=y, sr=sr)

    spec_image = spec_to_image(spectrogram=spectrogram)

    return spec_image
