import librosa
import numpy as np
from preprocessing.wave_to_spec import wave_to_spec
from preprocessing.spec_to_image import spec_to_image


def preprocess_sample(file_path: str) -> np.ndarray:
    """Preprocess audio file for CNN input.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        spec_image (np.ndarray): Preprocessed audio file as a spectrogram image.
    """    
    raw_audio, sr = librosa.load(file_path, sr=None)

    spectrogram = wave_to_spec(y=raw_audio, sr=sr)

    spec_image = spec_to_image(spectrogram=spectrogram)

    return spec_image
