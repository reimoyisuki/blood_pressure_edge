import torch
import torch.nn as nn

class CNN_LSTM_Attention(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Attention, self).__init__()
        
        # 1. CNN FEATURE EXTRACTOR
        # in_channels=2 karena menerima Sinyal PPG dan ECG sekaligus
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 2. RNN / LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        
        # 3. ATTENTION MECHANISM
        self.attn = nn.Linear(128, 1)  
        
        # 4. FULLY CONNECTED LAYER
        # Output 3 neuron: SBP, DBP, dan HR
        self.fc = nn.Linear(128, 3)    

    def forward(self, x):
        # Proses Utama untuk Training
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = x.permute(0, 2, 1)  
        lstm_out, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1) 
        out = self.fc(context) 
        
        return out

    def extract_features(self, x):
        """
        Fungsi khusus untuk mengekstrak bobot fitur dokumentasi laporan/JSON.
        Gunakan ini saat model sedang tidak di-train (fase evaluasi/testing).
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        x = x.permute(0, 2, 1)  
        lstm_out, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1) 
        predictions = self.fc(context)
        
        # Mengembalikan prediksi beserta fitur internalnya
        return {
            "predictions": predictions,
            "attention_weights": attn_weights,
            "context_vector": context
        }