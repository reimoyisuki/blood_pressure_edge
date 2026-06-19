import torch
import torch.nn as nn

class CNN_LSTM_Attention(nn.Module):
    def __init__(self):
        super(CNN_LSTM_Attention, self).__init__()
        
        # 1. CNN Feature Extractor (Murni 1 Input PPG)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, stride=1, padding=7),
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
        
        # 2. Temporal Extractor
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, bidirectional=True)
        
        # 3. Attention Mechanism
        self.attn = nn.Linear(128, 1)  
        
        # 4. Regression Head (SBP & DBP)
        self.fc = nn.Linear(128, 2)    

    def forward(self, x):
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
        Fungsi ini digunakan setelah training selesai, untuk mengtahui
        apa yang dilihat model (Attention Weights) dan intisari fiturnya (Context).
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.permute(0, 2, 1)  
        
        lstm_out, _ = self.lstm(x)
        
        # Ekstrak Bobot Perhatian (Titik fokus AI)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        
        # Ekstrak Fitur Laten Final (Intisari gelombang)
        context = torch.sum(attn_weights * lstm_out, dim=1) 
        
        predictions = self.fc(context)
        
        return predictions, attn_weights, context