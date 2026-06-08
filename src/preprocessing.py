import numpy as np
from scipy.signal import butter, filtfilt, detrend

def preprocess_ppg(ppg_signal, fs=125):
    """
    Membersihkan sinyal PPG dengan Bandpass Filter dan Baseline Correction
    """
    # 1. Bandpass filter 0.5–8 Hz untuk menghilangkan noise frekuensi tinggi/rendah
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)

    # 2. Baseline correction dengan detrend (biar sinyal gak melayang naik-turun)
    corrected = detrend(filtered, type='linear')

    # 3. Normalisasi (Z-score normalization)
    norm = (corrected - np.mean(corrected)) / (np.std(corrected) + 1e-8) # tambah 1e-8 biar ga error bagi nol
    
    return norm