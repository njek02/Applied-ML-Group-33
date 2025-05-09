import librosa
import numpy as np
import os
import matplotlib.pyplot as plt


def to_spectrogram(folder_path='data', limit=None, sr=None):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
    if limit:
        files = files[:limit]

    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        audio = np.load(file_path)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=256, hop_length=64)
    log_mel_spec = librosa.power_to_db(mel_spec)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_mel_spec, sr=sr, hop_length=64, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Log-Mel Spectrogram: {file_name}")
    plt.tight_layout()
    image_filename = os.path.splitext(file_name)[0] + '_spectrogram.png'
    image_path = os.path.join(folder_path, image_filename)
    plt.savefig(image_path)
    plt.close()