import os
import torch
import matplotlib.pyplot as plt
from src.dataset import get_data_loaders
from src.model import CNN_LSTM_Attention_Regressor
from src.engine import train_model

def plot_regression_results(train_losses, val_losses, train_maes, val_maes, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss (MSE)')
    plt.plot(epochs, val_losses, 'r-o', label='Val Loss (MSE)')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('MSE Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_maes, 'b-o', label='Train MAE')
    plt.plot(epochs, val_maes, 'r-o', label='Val MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.title('Mean Absolute Error (mmHg)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_regression_results.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    DATA_PATH = r"data/raw" 
    GDRIVE_MODEL_DIR = "/content/drive/MyDrive/Dataset_Magang/outputs"
    os.makedirs(GDRIVE_MODEL_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- MENGGUNAKAN DEVICE: {device.type.upper()} ---")
    
    train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, batch_size=32)
    
    model = CNN_LSTM_Attention_Regressor().to(device)
    
    print("\n--- Memulai Proses Training ---")
    train_loss, val_loss, train_mae, val_mae = train_model(
        model, train_loader, val_loader, device, epochs=100, lr=1e-3, output_dir=GDRIVE_MODEL_DIR
    )
    
    plot_regression_results(train_loss, val_loss, train_mae, val_mae, GDRIVE_MODEL_DIR)
    print(f"✅ Training Selesai! Model disimpan di folder {GDRIVE_MODEL_DIR}")