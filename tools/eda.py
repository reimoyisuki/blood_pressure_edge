import os
import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing import preprocess_ppg

def run_eda(data_path, patient_id):
    print(f"Menjalankan EDA untuk Pasien {patient_id}...")
    
    # Memuat file dari hasil ekstrak ppg.zip dan abp.zip
    ppg_fn = os.path.join(data_path, f"{patient_id}_ppg.npy")
    abp_fn = os.path.join(data_path, f"{patient_id}_abp.npy")
    
    if not os.path.exists(ppg_fn) or not os.path.exists(abp_fn):
        print(f"File data untuk pasien {patient_id} tidak ditemukan.")
        return
        
    ppg_data = np.load(ppg_fn)
    abp_data = np.load(abp_fn)
    
    # Ambil segmen pertama (index 0) sebagai sampel visualisasi
    segment_idx = 0
    raw_ppg = ppg_data[segment_idx]
    clean_ppg = preprocess_ppg(raw_ppg)
    raw_abp = abp_data[segment_idx]
    
    # Setup waktu (Sampling rate MIMIC-BP = 125 Hz)
    fs = 125
    t = np.arange(len(raw_ppg)) / fs
    
    # Ambil 5 detik pertama agar gelombang terlihat jelas (tidak berdempetan)
    t_5sec = t[t <= 5.0]
    
    plt.figure(figsize=(14, 8))
    
    # =========================================================
    # PLOT 1: Sinyal PPG Mentah (Input Asli)
    # =========================================================
    plt.subplot(3, 1, 1)
    plt.plot(t_5sec, raw_ppg[:len(t_5sec)], color='gray')
    plt.title(f'1. Sinyal PPG Mentah - Pasien {patient_id} (Input AI)', fontweight='bold')
    plt.ylabel('Amplitudo')
    plt.grid(True, alpha=0.3)
    
    # =========================================================
    # PLOT 2: Sinyal PPG Setelah Preprocessing (Input AI Final)
    # =========================================================
    plt.subplot(3, 1, 2)
    plt.plot(t_5sec, clean_ppg[:len(t_5sec)], color='green')
    plt.title('2. Sinyal PPG Setelah Preprocessing (Bandpass, Detrend, Clipping, Norm)', fontweight='bold')
    plt.ylabel('Z-Score')
    plt.grid(True, alpha=0.3)
    
    # =========================================================
    # PLOT 3: Sinyal ABP Mentah (Target / Ground Truth AI)
    # =========================================================
    plt.subplot(3, 1, 3)
    plt.plot(t_5sec, raw_abp[:len(t_5sec)], color='red')
    plt.title('3. Sinyal ABP Mentah (Target Prediksi / Ekstraksi SBP, DBP, HR)', fontweight='bold')
    plt.xlabel('Waktu (Detik)')
    plt.ylabel('Tekanan Darah (mmHg)')
    
    # Menandai secara visual di mana letak SBP (Sistolik) dan DBP (Diastolik)
    max_idx = np.argmax(raw_abp[:len(t_5sec)])
    min_idx = np.argmin(raw_abp[:len(t_5sec)])
    plt.plot(t_5sec[max_idx], raw_abp[max_idx], 'bo', label='Puncak Sistolik (SBP)')
    plt.plot(t_5sec[min_idx], raw_abp[min_idx], 'mo', label='Lembah Diastolik (DBP)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"eda_{patient_id}_signals.png"), dpi=300)
    print("✅ Grafik EDA berhasil disimpan!")
    plt.show()

if __name__ == "__main__":
    # Sesuaikan path ini dengan folder data_raw di Google Colab-mu
    DATA_PATH = r"/content/drive/MyDrive/Dataset_Magang/data_raw" 
    run_eda(DATA_PATH, "p093833")