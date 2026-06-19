import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3, 
                beta1=0.9, beta2=0.999, eps=1e-8, patience=10, output_dir="outputs"):
    
    # Menggunakan HuberLoss agar lebih tahan terhadap outlier tekanan darah ekstrem
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    patience_counter = 0
    history_log = []

    print("\n" + "="*80)
    print(f"{'EPOCH':^7} | {'TRAIN RMSE':^12} | {'VAL RMSE':^10} | {'TRAIN MAE':^10} | {'VAL MAE':^9} | {'VAL R2':^8}")
    print("="*80)

    for epoch in range(epochs):
        # ---------------- FASE TRAINING ----------------
        model.train()
        train_preds, train_targets = [], []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x) 
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_preds.append(outputs.detach().cpu().numpy())
            train_targets.append(y.detach().cpu().numpy())

        train_preds = np.vstack(train_preds)
        train_targets = np.vstack(train_targets)
        t_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
        t_mae = mean_absolute_error(train_targets, train_preds)
        t_r2 = r2_score(train_targets, train_preds)

        # ---------------- FASE VALIDASI ----------------
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

        # ---------------- SCHEDULING & LOGGING ----------------
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(v_rmse)
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f"{epoch+1:^7} | {t_rmse:^12.4f} | {v_rmse:^10.4f} | {t_mae:^10.4f} | {v_mae:^9.4f} | {v_r2:^8.4f}")
        
        if new_lr < current_lr:
            print(f"📉 [INFO] Val RMSE melambat. Learning Rate diturunkan menjadi: {new_lr:.2e}")

        history_log.append({
            "epoch": epoch + 1,
            "train_rmse": round(float(t_rmse), 4), "val_rmse": round(float(v_rmse), 4),
            "train_mae": round(float(t_mae), 4),   "val_mae": round(float(v_mae), 4),
            "train_r2": round(float(t_r2), 4),     "val_r2": round(float(v_r2), 4)
        })

        # ---------------- EARLY STOPPING ----------------
        os.makedirs(output_dir, exist_ok=True)
        if v_rmse < best_loss:
            best_loss = v_rmse
            patience_counter = 0 
            # Menyimpan model terbaik
            torch.save(model.state_dict(), os.path.join(output_dir, "bp_edge_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[INFO] Early stopping aktif pada epoch {epoch+1}. Model berhenti berlatih untuk mencegah overfitting.")
                break

    print("="*80)
    # Menyimpan riwayat metrik ke file JSON dan CSV
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history_log, f, indent=4)
        
    df_history = pd.DataFrame(history_log)
    df_history.to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    
    print(f"Riwayat metrik (JSON & CSV) tersimpan di: {output_dir}")
    return df_history