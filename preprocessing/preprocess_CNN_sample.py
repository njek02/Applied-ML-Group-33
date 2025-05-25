import librosa
import numpy as np
from PIL import Image


def preprocess_sample(file_path: str):
    y, sr = librosa.load(file_path, sr=None, duration=2.0)

    if sr != 2000:
        raise ValueError(f"Invalid sampling rate: {sr}. Expected 2000 Hz.")
    
    if len(y) < 4000:
        raise ValueError(f"Audio fragment too short: {len(y)/sr:.4f}. Minimum duration needs to be 2 seconds")

    spectrogram = wave_to_spec(y=y, sr=sr)

    spec_image = spec_to_image(spectrogram=spectrogram)

    return spec_image


def wave_to_spec(y, sr):
    # Apply Root Mean Square normalization
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        y = y / rms

    # Convert audio into a mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, hop_length=64, n_mels=64, fmin=100, fmax=400)
    db_mel_spec = librosa.power_to_db(mel_spec)

    return db_mel_spec


def spec_to_image(spectrogram: np.ndarray):
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