import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.dataset import get_data_loaders
from src.model import CNN_LSTM_Attention
from src.engine import train_model

def plot_results(df_history, output_dir):
    epochs = df_history['epoch']
    
    plt.figure(figsize=(15, 4))
    
    # 1. Plot RMSE
    plt.subplot(1, 3, 1)
    plt.plot(epochs, df_history['train_rmse'], 'b-', label='Train RMSE')
    plt.plot(epochs, df_history['val_rmse'], 'r-', label='Val RMSE')
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.legend(); plt.title('Root Mean Square Error')

    # 2. Plot MAE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, df_history['train_mae'], 'b-', label='Train MAE')
    plt.plot(epochs, df_history['val_mae'], 'r-', label='Val MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.title('Mean Absolute Error')

    # 3. Plot R2
    plt.subplot(1, 3, 3)
    plt.plot(epochs, df_history['train_r2'], 'b-', label='Train R2')
    plt.plot(epochs, df_history['val_r2'], 'r-', label='Val R2')
    plt.xlabel('Epoch'); plt.ylabel('R-Squared (R2)'); plt.legend(); plt.title('R-Squared')

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "training_grafik.png")
    plt.savefig(graph_path, dpi=300)
    print(f"📊 Grafik disimpan di: {graph_path}")
    plt.show()

if __name__ == "__main__":
    # --- PATH SESUAIKAN DENGAN GOOGLE DRIVE KAMU ---
    # Asumsikan kamu meletakkan dataset mentah di sini
    DATA_PATH = r"/content/drive/MyDrive/Dataset_Magang/data_raw" 
    
    # Output path yang kamu minta
    OUTPUT_DIR = r"/content/drive/MyDrive/Dataset_Magang/outputs/regresi"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Deteksi ketersediaan GPU T4 Google Colab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 MENGGUNAKAN DEVICE: {device.type.upper()} 🚀\n")
    
    print("--- 1. Memuat dan Memproses Data ---")
    # Jika datanya sangat besar, pastikan batch_size disesuaikan dengan VRAM GPU T4 (contoh: 32 atau 64)
    train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, batch_size=32, val_split=0.15, test_split=0.15)
    
    print("\n--- 2. Inisialisasi Model AI ---")
    model = CNN_LSTM_Attention().to(device)
    
    print("\n--- 3. Memulai Proses Training (100 Epochs) ---")
    # Kita menggunakan parameter Adam sesuai request
    df_history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device, 
        epochs=100, 
        lr=1e-3,          # Learning Rate
        beta1=0.9,        # Beta 1
        beta2=0.999,      # Beta 2
        eps=1e-8,         # Epsilon
        output_dir=OUTPUT_DIR
    )
    
    print("\n--- 4. Tabel Riwayat Training ---")
    # Menampilkan 5 epoch awal dan 5 epoch akhir
    print(df_history.head())
    print("...")
    print(df_history.tail())
    
    print("\n--- 5. Membuat Grafik Performa ---")
    plot_results(df_history, OUTPUT_DIR)
    
    print("\n✅ SELURUH PROSES SELESAI. Semua file tersimpan di Google Drive!")