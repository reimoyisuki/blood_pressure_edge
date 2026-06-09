import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """
        Save model when validation loss decrease.
        """
        import os
        os.makedirs("deployment", exist_ok=True)
        checkpoint_path = "deployment/bp_model.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f" -> Model terbaik ditemukan & disimpan! (Val Loss: {self.best_loss:.4f})")

def train_model(model, train_loader, val_loader, device, epochs=100, lr=1e-3):
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    for epoch in range(epochs):
        model.train()
        train_loss, total_mae_train = 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            mae = torch.mean(torch.abs(outputs - y)).item()
            total_mae_train += mae

        # Validasi
        model.eval()
        val_loss, total_mae_val = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - y)).item()
                total_mae_val += mae

        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(avg_val_loss)
        train_maes.append(total_mae_train / len(train_loader))
        val_maes.append(total_mae_val / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_losses[-1]:.4f}, Val Loss: {avg_val_loss:.4f} | "
              f"Train MAE: {train_maes[-1]:.4f}, Val MAE: {val_maes[-1]:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

    return train_losses, val_losses, train_maes, val_maes