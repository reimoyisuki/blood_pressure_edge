import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import preprocess_ppg

def run_eda(data_path, patient_id):
    print(f"--- Exploratory Data Analysis (EDA) untuk Pasien {patient_id} ---")
    
    ppg_fn = os.path.join(data_path, f"{patient_id}_ppg.npy")
    labels_fn = os.path.join(data_path, f"{patient_id}_labels.npy")
    
    if not os.path.exists(ppg_fn) or not os.path.exists(labels_fn):
        print("File data pasien tidak ditemukan di path tersebut!")
        return
        
    ppg_data = np.load(ppg_fn)
    labels_data = np.load(labels_fn) # Bentuk: (30, 2) -> SBP, DBP
    
    # ---------------------------------------------------------
    # EDA 1: Analisis Distribusi Label SBP & DBP
    # ---------------------------------------------------------
    sbp = labels_data[:, 0]
    dbp = labels_data[:, 1]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(sbp, kde=True, color='salmon', bins=10)
    plt.axvline(np.mean(sbp), color='red', linestyle='--', label=f'Mean: {np.mean(sbp):.1f}')
    plt.title('Distribusi SBP (Systolic Blood Pressure)')
    plt.xlabel('mmHg'); plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(dbp, kde=True, color='skyblue', bins=10)
    plt.axvline(np.mean(dbp), color='blue', linestyle='--', label=f'Mean: {np.mean(dbp):.1f}')
    plt.title('Distribusi DBP (Diastolic Blood Pressure)')
    plt.xlabel('mmHg'); plt.legend()
    
    plt.tight_layout()
    plt.savefig("eda_distribusi_label.png", dpi=300)
    plt.show()
    
    # ---------------------------------------------------------
    # EDA 2: Visualisasi Sinyal Mentah vs Preprocessed
    # ---------------------------------------------------------
    segment_idx = 0 # Ambil segmen pertama sebagai contoh
    raw_signal = ppg_data[segment_idx]
    processed_signal = preprocess_ppg(raw_signal)
    
    fs = 125 # Sampling rate
    t = np.arange(len(raw_signal)) / fs
    
    # Kita plot 5 detik pertama saja agar bentuk gelombangnya terlihat jelas
    t_5sec = t[t <= 5.0]
    
    plt.figure(figsize=(14, 6))
    
    # Sinyal Mentah
    plt.subplot(2, 1, 1)
    plt.plot(t_5sec, raw_signal[:len(t_5sec)], color='gray')
    plt.title(f'Sinyal PPG Mentah (Raw) - Pasien {patient_id} (5 Detik Pertama)')
    plt.ylabel('Amplitudo')
    plt.grid(True, alpha=0.3)
    
    # Sinyal Preprocessed
    plt.subplot(2, 1, 2)
    plt.plot(t_5sec, processed_signal[:len(t_5sec)], color='green')
    plt.title('Sinyal Setelah Bandpass Filter, Detrend, Artefact Removal & Z-Score')
    plt.xlabel('Waktu (detik)')
    plt.ylabel('Z-Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("eda_preprocessing_signal.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Ganti dengan path ke direktori data lokalmu
    DATA_PATH = r"data/raw" 
    run_eda(DATA_PATH, "p093833")