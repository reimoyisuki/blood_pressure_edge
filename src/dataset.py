import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.preprocessing import preprocess_ppg

class PPG_Edge_Dataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # Input: 1 Channel (PPG tunggal dari MAX30102)
        sig = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0)
        # Output/Target: 2 Nilai (SBP dan DBP)
        lbl = torch.tensor(self.labels[idx][:2], dtype=torch.float32) 
        return sig, lbl

def get_data_loaders(data_path, batch_size=32, val_split=0.15, test_split=0.15):
    # Mengambil ID unik pasien yang memiliki data PPG
    patients = set([fn.split("_")[0] for fn in os.listdir(data_path) if fn.endswith("_ppg.npy")])
    signals, labels = [], []

    print("Memuat dataset (Input: PPG, Target: SBP & DBP)...")
    for patient in sorted(list(patients)):
        ppg_fn = os.path.join(data_path, patient + "_ppg.npy")
        labels_fn = os.path.join(data_path, patient + "_labels.npy")
        
        if not os.path.exists(ppg_fn) or not os.path.exists(labels_fn): 
            continue
            
        ppg_data = np.load(ppg_fn)
        labels_data = np.load(labels_fn)

        # Proses iteratif untuk setiap segmen 30 detik
        for idx in range(len(ppg_data)):
            clean_ppg = preprocess_ppg(ppg_data[idx])
            signals.append(clean_ppg)
            # Hanya mengambil SBP (index 0) dan DBP (index 1)
            labels.append([labels_data[idx][0], labels_data[idx][1]])
            
    # Pembagian Data: 70% Train, 15% Validation, 15% Test
    X_temp, X_test, y_temp, y_test = train_test_split(signals, labels, test_size=test_split, random_state=42)
    val_ratio = val_split / (1.0 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

    # Pembuatan DataLoader
    train_loader = DataLoader(PPG_Edge_Dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PPG_Edge_Dataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PPG_Edge_Dataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader