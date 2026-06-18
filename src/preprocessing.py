import numpy as np
from scipy.signal import butter, filtfilt, detrend

def preprocess_ppg(ppg_signal, fs=125):
    # 1. Bandpass Filter (0.5 - 8 Hz)
    # Menghilangkan noise frekuensi rendah (napas) dan tinggi (noise listrik)
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)
    
    # 2. Baseline Correction
    # Menghilangkan efek wandering baseline (kemiringan sinyal)
    corrected = detrend(filtered, type='linear')
    
    # 3. Artefact Removal (Clipping)
    # Membuang lonjakan anomali ekstrem dengan membatasi di persentil 1% dan 99%
    lower_bound = np.percentile(corrected, 1)
    upper_bound = np.percentile(corrected, 99)
    clipped = np.clip(corrected, lower_bound, upper_bound)
    
    # 4. Normalisasi Z-Score
    # Mengubah skala sinyal menjadi standar deviasi 1 agar model DL mudah belajar
    norm = (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)
    
    return norm