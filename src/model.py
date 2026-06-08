import torch
import torch.nn as nn

class CNN_LSTM_Attention(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Attention, self).__init__()
        # CNN untuk ekstraksi fitur spasial/lokal dari gelombang
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM untuk ekstraksi pola urutan waktu (sekuensial)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        
        # Attention untuk fokus ke fitur gelombang yang paling penting
        self.attn = nn.Linear(128, 1)  
        
        # Output layer: 2 neuron (Sistolik dan Diastolik)
        self.fc = nn.Linear(128, 2)    

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)  # Ubah shape jadi (batch, seq_len, features) untuk LSTM
        
        lstm_out, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        out = self.fc(context)
        return out