import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Attention layer to calculate attention weights
        self.attention_layer = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, sequence_length, hidden_size)
        attention_scores = torch.tanh(self.attention_layer(lstm_output))  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, sequence_length, 1)

        # context vector: weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size)

        return context_vector, attention_weights


class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
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