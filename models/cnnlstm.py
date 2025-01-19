import torch
import torch.nn as nn

from .base import BaseModel

class CNNLSTMModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CNNLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(64, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Adjust input for Conv1D: [batch_size, input_size, sequence_length]
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.permute(0, 2, 1)  # Back to [batch_size, sequence_length, features]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(self.dropout(out[:, -1, :]))
        return out

    def predict(self, x):
        """
        Perform prediction with the CNN-LSTM model.
        """
        self.eval()  # Evaluation mode
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32).to(next(self.parameters()).device)
            outputs = self.forward(x)
            return outputs.cpu().numpy()