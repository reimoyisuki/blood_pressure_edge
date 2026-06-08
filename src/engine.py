import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_maes, val_maes = [], []

    for epoch in range(epochs):
        # Tahap Training
        model.train()
        train_loss, total_mae_train = 0, 0
        
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            mae = torch.mean(torch.abs(outputs - y)).item()
            total_mae_train += mae

        # Tahap Validasi
        model.eval()
        val_loss, total_mae_val = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                mae = torch.mean(torch.abs(outputs - y)).item()
                total_mae_val += mae

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_train_mae = total_mae_train / len(train_loader)
        avg_val_mae = total_mae_val / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_val_mae)

        print(f"Epoch {epoch+1}/{epochs} | "f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | "f"Train MAE: {avg_train_mae:.4f}, Val MAE: {avg_val_mae:.4f}")

    return train_losses, val_losses, train_maes, val_maes