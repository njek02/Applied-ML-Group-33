import librosa
from preprocessing.normalization import rms_normalize


def wave_to_spec(y, sr):

    y = rms_normalize(y)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=256, hop_length=64, n_mels=64, fmin=100, fmax=400)

    db_mel_spec = librosa.power_to_db(mel_spec)

    return db_mel_spec
