import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, 
                beta1=0.9, beta2=0.999, eps=1e-8, output_dir="deployment"):
    
    # Kriteria Loss (Bisa pakai MSE atau Huber Loss)
    criterion = nn.MSELoss()
    
    # 1. Adam Optimizer dengan Parameter Kustom
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    
    # 2. Learning Rate Scheduling (Turunkan LR setengahnya jika Val RMSE tidak membaik selama 5 epoch)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    history_log = []

    print("\n" + "="*80)
    print(f"{'EPOCH':^7} | {'TRAIN RMSE':^12} | {'VAL RMSE':^10} | {'TRAIN MAE':^10} | {'VAL MAE':^9} | {'VAL R2':^8}")
    print("="*80)

    for epoch in range(epochs):
        # ==========================================
        # FASE TRAINING
        # ==========================================
        model.train()
        train_preds, train_targets = [], []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x) # Linear activation otomatis terjadi di layer nn.Linear
            
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            # Simpan hasil untuk metrik epoch-wise
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(y.detach().cpu().numpy())

        # Gabungkan semua batch untuk akurasi metrik yang valid
        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        
        t_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        t_mae = mean_absolute_error(train_targets, train_preds)
        t_r2 = r2_score(train_targets, train_preds)

        # ==========================================
        # FASE VALIDATION
        # ==========================================
        model.eval()
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(y.cpu().numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        
        v_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
        v_mae = mean_absolute_error(val_targets, val_preds)
        v_r2 = r2_score(val_targets, val_preds)

        # ==========================================
        # LOGGING & SCHEDULING
        # ==========================================
        scheduler.step(v_rmse)
        
        print(f"{epoch+1:^7} | {t_rmse:^12.4f} | {v_rmse:^10.4f} | {t_mae:^10.4f} | {v_mae:^9.4f} | {v_r2:^8.4f}")

        history_log.append({
            "epoch": epoch + 1,
            "train_rmse": round(t_rmse, 4), "val_rmse": round(v_rmse, 4),
            "train_mae": round(t_mae, 4),   "val_mae": round(v_mae, 4),
            "train_r2": round(t_r2, 4),     "val_r2": round(v_r2, 4)
        })

        # Save Best Model
        os.makedirs(output_dir, exist_ok=True)
        if v_rmse < best_loss:
            best_loss = v_rmse
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "bp_multimodal_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping aktif pada epoch {epoch+1} karena tidak ada perbaikan.")
                break

    print("="*80)
    
    # Simpan history ke JSON
    json_path = os.path.join(output_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history_log, f, indent=4)
        
    # Simpan history ke CSV
    csv_path = os.path.join(output_dir, "training_history.csv")
    df_history = pd.DataFrame(history_log)
    df_history.to_csv(csv_path, index=False)
    
    print(f"📄 Riwayat metrik (JSON & CSV) tersimpan di: {output_dir}")

    return df_history