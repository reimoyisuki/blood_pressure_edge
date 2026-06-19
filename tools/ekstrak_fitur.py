import os
import torch
import numpy as np
import pandas as pd
import json
from src.model import CNN_LSTM_Attention
from src.preprocessing import preprocess_ppg

def save_features_to_disk(data_path, model_path, output_dir, patient_id):
    print(f"Mengekstrak fitur dari pasien {patient_id}...")
    
    # 1. Setup Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_LSTM_Attention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Ambil Sinyal PPG Pasien
    ppg_fn = os.path.join(data_path, f"{patient_id}_ppg.npy")
    raw_signal = np.load(ppg_fn)[0] # Ambil segmen pertama
    clean_signal = preprocess_ppg(raw_signal)
    
    # Konversi ke Tensor dengan bentuk (Batch=1, Channel=1, Panjang=3750)
    input_tensor = torch.tensor(clean_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Ekstrak Fitur menggunakan fungsi baru
    with torch.no_grad():
        preds, attn, context = model.extract_features(input_tensor)
        
    # Tarik data dari memori GPU ke CPU agar bisa di-save
    preds_np = preds.cpu().numpy()[0]
    attn_np = attn.cpu().numpy()[0].flatten() # Jadikan array 1D
    context_np = context.cpu().numpy()[0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # SIMPAN KE CSV (Cocok untuk laporan/tabel Excel)
    df_attn = pd.DataFrame({
        "time_step": np.arange(len(attn_np)),
        "attention_weight": attn_np
    })
    csv_path = os.path.join(output_dir, f"{patient_id}_attention_features.csv")
    df_attn.to_csv(csv_path, index=False)
    
    # SIMPAN KE JSON
    feature_dict = {
        "patient_id": patient_id,
        "predictions": {
            "SBP": float(preds_np[0]),
            "DBP": float(preds_np[1])
        },
        "context_vector": context_np.tolist(), # 128 angka intisari sinyal
        "attention_weights": attn_np.tolist()  # Bobot tiap titik waktu
    }
    
    json_path = os.path.join(output_dir, f"{patient_id}_extracted_features.json")
    with open(json_path, "w") as f:
        json.dump(feature_dict, f, indent=4)
        
    print(f"Ekstraksi selesai! File tersimpan di: {output_dir}")

if __name__ == "__main__":
    DATA_PATH = r"/content/drive/MyDrive/Dataset_Magang/data_raw"
    MODEL_PATH = r"/content/drive/MyDrive/Dataset_Magang/outputs/regresi/bp_edge_model.pth"
    OUTPUT_DIR = r"/content/drive/MyDrive/Dataset_Magang/outputs/ekstraksi_fitur"
    
    # Ekstrak fitur dari salah satu pasien
    save_features_to_disk(DATA_PATH, MODEL_PATH, OUTPUT_DIR, "p093833")