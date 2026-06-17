import numpy as np
from scipy.signal import butter, filtfilt, detrend

def preprocess_ppg(ppg_signal, fs=125):
    # Bandpass Filter 0.5-8 Hz
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)
    
    # Baseline Correction (Artefact Removal)
    corrected = detrend(filtered, type='linear')
    
    # Normalisasi Z-Score
    norm = (corrected - np.mean(corrected)) / (np.std(corrected) + 1e-8)
    return norm