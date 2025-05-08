import numpy as np
import librosa


def rms_normalize(y):
    rms = np.sqrt(np.mean(y**2))
    if rms > 0:
        return y / rms
    else:
        return y


def wave_to_spec(y, sr, n_fft, hop_len, n_mels):

    y = rms_normalize(y)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                              hop_length=hop_len,
                                              n_mels=n_mels)

    db_mel_spec = librosa.power_to_db(mel_spec)

    return db_mel_spec
