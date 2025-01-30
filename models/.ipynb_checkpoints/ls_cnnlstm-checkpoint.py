import torch
import torch.nn as nn


class LongShortCNNLSTM(nn.Module):
    def __init__(self, 
                 long_input_size, 
                 short_input_size, 
                 hidden_size,  
                 num_layers, 
                 long_output_size, 
                 short_output_size, 
                 long_term_length, 
                 short_term_length, 
                 dropout=0.3):
        super(LongShortCNNLSTM, self).__init__()

        self.dropout_rate = dropout

        # Long-term CNN
        self.long_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, long_input_size), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )

        self.long_cnn_output_size = self._calculate_cnn_output_size(
            self.long_cnn, (1, 1, long_term_length, long_input_size)
        )


        self.long_lstm = nn.LSTM(
            input_size=self.long_cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )

        # Short-term CNN
        self.short_cnn = nn.Sequential(
            nn.Conv1d(short_input_size, 16, kernel_size=3, stride=1, padding=1),  # Use dynamic input size
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.short_cnn_output_size = self._calculate_cnn_output_size(
            self.short_cnn, (1, short_input_size, short_term_length)
        )

        self.short_lstm = nn.LSTM(
            input_size=self.short_cnn_output_size + hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )

        self.long_fc = nn.Linear(hidden_size * 2, long_output_size)
        self.short_fc = nn.Linear(hidden_size * 2, short_output_size)

    def forward(self, long_input, short_input):
        # Long-term processing
        long_features = self.long_cnn(long_input.unsqueeze(1))
        long_features = long_features.view(long_features.size(0), long_features.size(2), -1)
        long_output, _ = self.long_lstm(long_features)
        
        long_output = long_output[:, -1, :]
        long_final = self.long_fc(long_output)

        # Short-term processing
        short_features = self.short_cnn(short_input.permute(0, 2, 1))
        short_features = short_features.view(short_features.size(0), short_features.size(2), -1)
        
        combined_features = torch.cat(
            (short_features, long_output.unsqueeze(1).repeat(1, short_features.size(1), 1)), dim=2
        )
        short_output, _ = self.short_lstm(combined_features)
        short_final = self.short_fc(short_output[:, -1, :])

        return long_final, short_final


    def _calculate_cnn_output_size(self, module, input_shape):
        """
        Calculate the effective LSTM input size after CNN layers.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)  # Create dummy input
            output = module(dummy_input)  # Pass through CNN layers
            if len(output.shape) == 4:  # Conv2d
                return output.size(1) * output.size(3)
            elif len(output.shape) == 3:  # Conv1d
                return output.size(1) 