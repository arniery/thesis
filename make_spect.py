import os
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)

# Constants
mel_basis = mel(sr=16000, n_fft=1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

# Directories
rootDir = '/Users/arniexx/Desktop/thesis/dialect-tts/autovc/hindienglish'
targetDir = './spmel'

# Process each speaker folder
for speaker_id in sorted(os.listdir(rootDir)):
    speaker_path = os.path.join(rootDir, speaker_id)
    if not os.path.isdir(speaker_path):
        continue  # skip files

    # Create corresponding folder in spmel
    speaker_target = os.path.join(targetDir, speaker_id)
    os.makedirs(speaker_target, exist_ok=True)

    for wav_file in sorted(os.listdir(speaker_path)):
        if not wav_file.endswith('.wav'):
            continue

        wav_path = os.path.join(speaker_path, wav_file)
        x, fs = sf.read(wav_path)
        prng = RandomState(int(abs(hash(wav_file)) % (10**8)))  # seed

        # Preprocess
        y = signal.filtfilt(b, a, x)
        wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-6
        D = pySTFT(wav).T
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1)

        # Save .npy
        save_name = wav_file.replace('.wav', '.npy')
        save_path = os.path.join(speaker_target, save_name)
        np.save(save_path, S.astype(np.float32), allow_pickle=False)
        print(f"Saved: {save_path}")

# ========== new make spect.py ==========
import os
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a
    
    
def pySTFT(x, fft_length=1024, hop_length=256):
    
    x = np.pad(x, int(fft_length//2), mode='reflect')
    
    noverlap = fft_length - hop_length
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
    strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                             strides=strides)
    
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    
    return np.abs(result)    
    
import librosa
mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80, fmin=90, fmax=7600).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)


# audio file directory
rootDir = './wavs'
# spectrogram directory
targetDir = './spmel'


dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

for subdir in sorted(subdirList):
    print(subdir)
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    _,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
    prng = RandomState(int(subdir[1:])) 
    for fileName in sorted(fileList):
        # Read audio file
        x, fs = sf.read(os.path.join(dirName,subdir,fileName))
        # Remove drifting noise
        y = signal.filtfilt(b, a, x)
        # Ddd a little random noise for model roubstness
        # wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
        # Compute spect
        D = pySTFT(y).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel))
        # S = np.clip((D_db + 100) / 100, 0, 1)    
        # save spect    
        np.save(os.path.join(targetDir, subdir, fileName[:-4]),
                D_db.astype(np.float32), allow_pickle=False)    
        
  
        
