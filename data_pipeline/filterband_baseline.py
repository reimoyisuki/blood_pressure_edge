import os
import numpy as np
from scipy.signal import butter, filtfilt, detrend
import matplotlib.pyplot as plt

def load_mimic_bp_data(dbPath, patient, idx_list):
    """
    Membaca sinyal PPG dan label SBP/DBP dari MIMIC-BP
    dbPath  : folder tempat file .npy
    patient : ID pasien (misalnya 'p074514')
    idx_list: daftar indeks segmen yang ingin dipakai
    """
    # load PPG
    ppg_fn = patient + "_ppg.npy"
    ppg_data = np.load(os.path.join(dbPath, ppg_fn))   # shape (30, 3750)

    # load labels (SBP, DBP)
    labels_fn = patient + "_labels.npy"
    labels = np.load(os.path.join(dbPath, labels_fn))  # shape (30, 2)

    signals, targets = [], []
    for idx in idx_list:
        sig = preprocess_ppg(ppg_data[idx])   # preprocessing sinyal PPG
        signals.append(sig)
        targets.append(labels[idx])           # [SBP, DBP]

    return signals, targets

# -----------------------------
# 1. Preprocessing + Baseline Correction
# -----------------------------
def preprocess_ppg(ppg_signal, fs=125):
    # bandpass filter 0.5–8 Hz untuk menghilangkan noise
    b, a = butter(3, [0.5/(fs/2), 8/(fs/2)], btype='band')
    filtered = filtfilt(b, a, ppg_signal)

    # baseline correction dengan detrend
    corrected = detrend(filtered, type='linear')

    # normalisasi
    norm = (corrected - np.mean(corrected)) / np.std(corrected)
    return norm

# -----------------------------
# Contoh penggunaan
# -----------------------------
if __name__ == "__main__":
    # contoh sinyal dummy (ganti dengan sinyal PPG asli)
   # t = np.linspace(0, 10, 1250)  # 10 detik @125 Hz
    # ppg_signal = np.sin(2*np.pi*1.2*t) + 0.5*np.sin(2*np.pi*0.1*t)  # sinyal + baseline drift

   #  processed = preprocess_ppg(ppg_signal)


parser = argparse.ArgumentParser(
        prog="read_data",
        description="How to read files from MIMIC-BP",
        epilog="Last update: 20Jun2023",
    )

parser.add_argument(
        "-d", "--dbPath", help="path to .npy files", required=True
    )
    parser.add_argument(
        "-p", "--patient", help="patient ID (eg, p093833)", required=True
    )
    parser.add_argument(
        "-i", "--idx", type=int, help="segment index", required=True
    )
    parser.add_argument(
        "-g", "--graph", action="store_true", help="flag for graphs"
    )
    args = parser.parse_args()

    showBP(args.dbPath, args.patient, args.idx)




    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,5))
    plt.subplot(2,1,1)
    plt.plot(t, ppg_signal)
    plt.title("Sinyal PPG Asli (dengan baseline drift)")
    plt.subplot(2,1,2)
    plt.plot(t, processed)
    plt.title("Sinyal PPG setelah Baseline Correction & Normalisasi")
    plt.tight_layout()
    plt.show()
