import os
import torch
import matplotlib.pyplot as plt
from src.dataset import get_data_loaders
from src.model import CNN_LSTM_Attention
from src.engine import train_model

def plot_results(train_losses, val_losses, train_maes, val_maes):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 4))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss (MSE)')
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss (MSE)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_maes, 'b-o', label='Train MAE')
    plt.plot(epochs, val_maes, 'r-o', label='Val MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.title('Mean Absolute Error')

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=300)
    print("Grafik disimpan sebagai 'training_results.png'")
    plt.show()

if __name__ == "__main__":
    DATA_PATH = r"data/raw" 
    DEPLOY_DIR = r"deployment"
    os.makedirs(DEPLOY_DIR, exist_ok=True)
    
    # Deteksi ketersediaan GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- MENGGUNAKAN DEVICE: {device.type.upper()} ---\n")
    
    print("--- Memuat dan Memproses Data ---")
    train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, batch_size=16, val_split=0.2)
    
    print("\n--- Inisialisasi Model AI ---")
    # Pindahkan seluruh arsitektur model ke GPU
    model = CNN_LSTM_Attention().to(device)
    
    print("\n--- Memulai Proses Training ---")
    # Lempar variabel device ke dalam fungsi train_model
    train_loss, val_loss, train_mae, val_mae = train_model(
        model, train_loader, val_loader, device, epochs=100, lr=1e-3
    )
    
    print("\n--- Menyimpan Model ---")
    save_path = os.path.join(DEPLOY_DIR, "bp_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model berhasil disimpan ke: {save_path}")
    
    print("\n--- Membuat Grafik Performa ---")
    plot_results(train_loss, val_loss, train_mae, val_mae)