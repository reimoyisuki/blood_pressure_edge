import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json  

def calculate_r2(y_true, y_pred):
    target_mean = torch.mean(y_true, dim=0)
    ss_tot = torch.sum((y_true - target_mean) ** 2, dim=0)
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    return torch.mean(r2).item()

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, output_dir="deployment"):
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # LR Scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    train_losses, val_losses, train_maes, val_maes = [], [], [], []
    history_log = [] # 1. Wadah untuk menampung metrik dari epoch 1 sampai akhir

    for epoch in range(epochs):
        model.train()
        train_mse, train_mae, train_r2 = 0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_mse += loss.item()
            train_mae += torch.mean(torch.abs(outputs - y)).item()
            train_r2 += calculate_r2(y, outputs)

        model.eval()
        val_mse, val_mae, val_r2 = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                val_mse += criterion(outputs, y).item()
                val_mae += torch.mean(torch.abs(outputs - y)).item()
                val_r2 += calculate_r2(y, outputs)

        # Kalkulasi nilai rata-rata metrik per epoch
        t_rmse = np.sqrt(train_mse / len(train_loader))
        t_mae = train_mae / len(train_loader)
        
        v_rmse = np.sqrt(val_mse / len(val_loader))
        v_mae = val_mae / len(val_loader)
        v_r2 = val_r2 / len(val_loader)

        train_losses.append(train_mse / len(train_loader))
        val_losses.append(val_mse / len(val_loader))
        train_maes.append(t_mae)
        val_maes.append(v_mae)

        history_log.append({
            "epoch": epoch + 1,
            "train_loss_mse": round(train_mse / len(train_loader), 4),
            "val_loss_mse": round(val_mse / len(val_loader), 4),
            "train_rmse": round(t_rmse, 4),
            "val_rmse": round(v_rmse, 4),
            "train_mae": round(t_mae, 4),
            "val_mae": round(v_mae, 4),
            "val_r2": round(v_r2, 4)
        })

        scheduler.step(v_rmse)
        print(f"Epoch {epoch+1}/{epochs} | Tr RMSE: {t_rmse:.2f} | Val RMSE: {v_rmse:.2f} | Val MAE: {v_mae:.2f} | Val R2: {v_r2:.4f}")

        # Simpan checkpoint model terbaik jika Val RMSE mengalami penurunan
        if v_rmse < best_loss:
            best_loss = v_rmse
            patience_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "bp_regressor_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping aktif pada epoch {epoch+1}!")
                break

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history_log, f, indent=4)
    print(f"📄 Berkas riwayat pelatihan JSON berhasil disimpan ke: {json_path}")

    return train_losses, val_losses, train_maes, val_maes