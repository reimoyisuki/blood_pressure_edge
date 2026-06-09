import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import preprocess_ppg

class PPGDataset(Dataset):
    def __init__(self, signals, labels):
        self.signals = signals
        self.labels = labels

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        # PyTorch CNN1D butuh format (batch, channel, seq_len) -> channel=1
        signal = torch.tensor(self.signals[idx], dtype=torch.float32).unsqueeze(0) 
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # (SBP, DBP)
        return signal, label

def get_data_loaders(data_path, batch_size=16, val_split=0.15, test_split=0.15):
    """
    Fungsi untuk membaca file .npy, preprocessing, dan membagi train/val/test loader
    """
    patients = set([fn.split("_")[0] for fn in os.listdir(data_path) if fn.endswith("_ppg.npy")])
    signals, labels = [], []

    for patient in patients:
        ppg_fn = os.path.join(data_path, patient + "_ppg.npy")
        labels_fn = os.path.join(data_path, patient + "_labels.npy")
        
        if not os.path.exists(ppg_fn) or not os.path.exists(labels_fn):
            continue
            
        ppg_data = np.load(ppg_fn)
        labels_data = np.load(labels_fn)

        if len(ppg_data) == 0 or len(labels_data) == 0:
            continue

        # Gabungkan dan preprocess semua segmen
        for idx in range(len(ppg_data)):
            processed = preprocess_ppg(ppg_data[idx])
            signals.append(processed)
            labels.append(labels_data[idx])

    print(f"Total sinyal berhasil dimuat: {len(signals)}")
    
    # Menghitung indeks pemotongan (70% Train, 15% Val, 15% Test)
    total_data = len(signals)
    train_ratio = 1.0 - val_split - test_split # Menjadi 0.70
    
    train_split_idx = int(train_ratio * total_data)
    val_split_idx = int(val_split * total_data)

    # 1. Potong Data Training
    train_signals = signals[:train_split_idx]
    train_labels = labels[:train_split_idx]

    # 2. Potong Data Validation
    val_signals = signals[train_split_idx : train_split_idx + val_split_idx]
    val_labels = labels[train_split_idx : train_split_idx + val_split_idx]

    # 3. Potong Data Testing (Sisa 15% terakhir)
    test_signals = signals[train_split_idx + val_split_idx :]
    test_labels = labels[train_split_idx + val_split_idx :]

    # Masukkan ke format Dataset
    train_dataset = PPGDataset(train_signals, train_labels)
    val_dataset = PPGDataset(val_signals, val_labels)
    test_dataset = PPGDataset(test_signals, test_labels)

    # Bungkus ke DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Test loader tidak perlu di-shuffle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Kembalikan 3 loader
    return train_loader, val_loader, test_loader