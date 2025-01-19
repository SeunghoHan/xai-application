import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Attention layer to calculate attention weights
        self.attention_layer = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.randn(hidden_size) * 0.1)  # Improved initialization

    def forward(self, lstm_output):
        attention_scores = torch.tanh(self.attention_layer(lstm_output))  # (batch_size, sequence_length, hidden_size)
        attention_scores = torch.matmul(attention_scores, self.v)  # (batch_size, sequence_length)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)  # (batch_size, sequence_length, 1)

        # context vector: weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size)
        return context_vector, attention_weights


class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=24, dropout=0.3):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with dropout applied if num_layers > 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = Attention(hidden_size)
        
        # Fully connected layer for output prediction
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM forward pass
        lstm_output, _ = self.lstm(x)  # lstm_output: (batch_size, sequence_length, hidden_size)
        
        # Apply attention mechanism
        context_vector, attn_weights = self.attention(lstm_output)

        # Pass through dropout and fully connected layer
        output = self.fc(self.dropout(context_vector))  # (batch_size, output_size)

        return output, attn_weights.squeeze(-1)


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, output_size=24, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Attention Layer
        self.attention = Attention(hidden_size)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(x, (h0, c0))  # lstm_output: (batch_size, sequence_length, hidden_size)
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(lstm_output)  # (batch_size, hidden_size)

        # Final prediction
        output = self.fc(self.dropout(context_vector))  # (batch_size, output_size)
        return output, attention_weights




class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=3, output_size=24, dropout=0.3):
        super(BiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # Attention Layer
        self.attention = Attention(hidden_size * 2)  # Bidirectional이므로 hidden_size가 2배가 됨.

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Initialize hidden state and cell state for bidirectional LSTM
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2배 된 hidden size
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_output, _ = self.lstm(x, (h0, c0))  # lstm_output: (batch_size, sequence_length, hidden_size*2)
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(lstm_output)  # (batch_size, hidden_size*2)

        # Final prediction
        output = self.fc(self.dropout(context_vector))  # (batch_size, output_size)
        return output, attention_weights
