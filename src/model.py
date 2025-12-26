import torch
import torch.nn as nn

class LSTMEmotionModel(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=8):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Expecting (batch, features)
        # Convert to (batch, seq_len=1, features)
        x = x.unsqueeze(1)

        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out
