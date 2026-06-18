import os
import torch
import numpy as np
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_ppg

# Fungsi baru untuk ekstrak Heart Rate dari ABP
def extract_hr_from_abp(abp_signal, fs=125):
    # Cari puncak (sistolik) pada gelombang ABP
    peaks, _ = find_peaks(abp_signal, distance=fs*0.4, height=np.mean(abp_signal))
    jumlah_detak = len(peaks)
    # Segmen 30 detik, jadi dikali 2 untuk dapat Beats Per Minute (BPM)
    return float(jumlah_detak * 2)

class Multimodal_Regression_Dataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Format input (Channel=2 (PPG & ECG), Sequence Length=3750)
        sig = torch.tensor(self.signals[idx], dtype=torch.float32)
        # Target = SBP, DBP, HR (3 label)
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32) 
        return sig, lbl

def get_data_loaders(data_path, batch_size=32, val_split=0.15, test_split=0.15):
    patients = set([fn.split("_")[0] for fn in os.listdir(data_path) if fn.endswith("_ppg.npy")])
    signals, labels = [], []

    print("Memproses dataset (PPG, ECG, ABP, Labels)...")
    for patient in sorted(list(patients)):
        ppg_fn = os.path.join(data_path, patient + "_ppg.npy")
        ecg_fn = os.path.join(data_path, patient + "_ecg.npy")
        abp_fn = os.path.join(data_path, patient + "_abp.npy")
        labels_fn = os.path.join(data_path, patient + "_labels.npy")
        
        # Pastikan 4 file ini ada untuk setiap pasien
        if not all(os.path.exists(f) for f in [ppg_fn, ecg_fn, abp_fn, labels_fn]):
            continue
            
        ppg_data = np.load(ppg_fn)
        ecg_data = np.load(ecg_fn)
        abp_data = np.load(abp_fn)
        labels_data = np.load(labels_fn) # SBP, DBP

        for idx in range(len(ppg_data)):
            # 1. Preprocessing Sinyal Input
            # Kita bisa pakai fungsi preprocess_ppg untuk membersihkan ECG juga
            clean_ppg = preprocess_ppg(ppg_data[idx])
            clean_ecg = preprocess_ppg(ecg_data[idx]) 
            
            # Tumpuk (Stack) PPG dan ECG jadi 2 Channel -> bentuk: (2, 3750)
            combined_signal = np.vstack([clean_ppg, clean_ecg])
            signals.append(combined_signal)
            
            # 2. Pembuatan Label (Target)
            # Ekstrak HR dari ABP mentah
            hr_val = extract_hr_from_abp(abp_data[idx])
            sbp_val = labels_data[idx][0]
            dbp_val = labels_data[idx][1]
            
            # Gabungkan jadi [SBP, DBP, HR]
            combined_labels = [sbp_val, dbp_val, hr_val]
            labels.append(combined_labels)
            
    # Data Split (70% Train, 15% Val, 15% Test)
    X_temp, X_test, y_temp, y_test = train_test_split(signals, labels, test_size=test_split, random_state=42)
    val_ratio = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

    train_loader = DataLoader(Multimodal_Regression_Dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Multimodal_Regression_Dataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Multimodal_Regression_Dataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader