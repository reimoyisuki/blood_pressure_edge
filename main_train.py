import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.dataset import get_data_loaders
from src.model import CNN_LSTM_Attention
from src.engine import train_model

def plot_results(df_history, output_dir):
    epochs = df_history['epoch']
    plt.figure(figsize=(15, 5))
    
    # Grafik RMSE
    plt.subplot(1, 3, 1)
    plt.plot(epochs, df_history['train_rmse'], 'b-o', markersize=3, label='Train')
    plt.plot(epochs, df_history['val_rmse'], 'r-o', markersize=3, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('RMSE'); plt.legend(); plt.title('Root Mean Square Error')
    plt.grid(alpha=0.3)

    # Grafik MAE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, df_history['train_mae'], 'b-o', markersize=3, label='Train')
    plt.plot(epochs, df_history['val_mae'], 'r-o', markersize=3, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MAE (mmHg)'); plt.legend(); plt.title('Mean Absolute Error')
    plt.grid(alpha=0.3)

    # Grafik R-Squared
    plt.subplot(1, 3, 3)
    plt.plot(epochs, df_history['train_r2'], 'b-o', markersize=3, label='Train')
    plt.plot(epochs, df_history['val_r2'], 'r-o', markersize=3, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('R2 Score'); plt.legend(); plt.title('R-Squared')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    graph_path = os.path.join(output_dir, "training_grafik_edge.png")
    plt.savefig(graph_path, dpi=300)
    print(f"Grafik divisualisasikan dan disimpan di: {graph_path}")
    plt.show()

def evaluate_test_data(model, test_loader, device, model_path):
    print("\n" + "="*80)
    print("MEMULAI FASE TESTING (EVALUASI PADA DATA UNSEEN) ")
    print("="*80)
    
    # Memuat bobot model terbaik hasil dari fase training
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Kunci model agar tidak melakukan update bobot
    
    test_preds, test_targets = [], []
    
    # Mematikan komputasi gradien untuk menghemat memori
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
            test_preds.append(outputs.cpu().numpy())
            test_targets.append(y.cpu().numpy())
            
    # Menggabungkan hasil seluruh batch
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    
    # Menghitung Metrik Final
    t_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
    t_mae = mean_absolute_error(test_targets, test_preds)
    t_r2 = r2_score(test_targets, test_preds)
    
    print(f"HASIL AKHIR PADA DATA TESTING:")
    print(f"   - Test RMSE : {t_rmse:.4f}")
    print(f"   - Test MAE  : {t_mae:.4f} mmHg")
    print(f"   - Test R2   : {t_r2:.4f}")
    print("="*80)

if __name__ == "__main__":
    # Setup Path yang mengarah ke Google Drive
    DATA_PATH = r"/content/drive/MyDrive/Dataset_Magang/data_raw" 
    OUTPUT_DIR = r"/content/drive/MyDrive/Dataset_Magang/outputs/regresi"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMEMULAI PROSES DI DEVICE: {device.type.upper()}\n")
    
    # 1. Menyiapkan DataLoader (Train, Val, dan Test)
    train_loader, val_loader, test_loader = get_data_loaders(DATA_PATH, batch_size=32)
    
    # 2. Inisialisasi Model
    model = CNN_LSTM_Attention().to(device)
    
    # 3. Menjalankan Engine Training (Train & Validasi)
    df_history = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device, 
        epochs=100, 
        lr=1e-3, 
        patience=10, 
        output_dir=OUTPUT_DIR
    )
    
    # 4. Evaluasi Visual
    plot_results(df_history, OUTPUT_DIR)
    
    # 5. Fase Testing
    best_model_path = os.path.join(OUTPUT_DIR, "bp_edge_model.pth")
    evaluate_test_data(model, test_loader, device, best_model_path)
    
    print("\nSELURUH PROSES SELESAI!")