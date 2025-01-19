import torch
import torch.nn as nn

from .base import BaseModel

class MultiTermCNNLSTM(BaseModel):
    def __init__(self, input_size, long_term_days, short_term_days, hidden_size, num_layers, output_size):
        super(MultiTermCNNLSTM, self).__init__()

        self.dropout_rate = 0.3

        # Long-term CNN
        self.long_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, input_size), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Conv2d(32, 64, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(3, 1))
        )
        
        # Short-term CNN
        self.short_cnn = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(63),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Flattened feature sizes
        self.long_cnn_out_size = self._calculate_cnn_output_size(self.long_cnn, (1, 1, long_term_days, input_size))
        self.short_cnn_out_size = self._calculate_cnn_output_size(self.short_cnn, (1, input_size, short_term_days))
        self.flattened_features = self.long_cnn_out_size + self.short_cnn_out_size

        # Fully connected fusion layer
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.flattened_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=self.dropout_rate,
            bidirectional=True
        )
        
        # Fully connected output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size),
            nn.Sigmoid()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, long_input, short_input):
        long_features = self.long_cnn(long_input.unsqueeze(1))
        long_features = long_features.view(long_features.size(0), -1)
        short_features = self.short_cnn(short_input.permute(0, 2, 1))
        short_features = short_features.view(short_features.size(0), -1)
        fused_features = torch.cat((long_features, short_features), dim=1)
        fused_features = self.fusion_fc(fused_features)
        lstm_input = fused_features.unsqueeze(1)
        lstm_output, _ = self.lstm(lstm_input)
        output = self.fc(lstm_output[:, -1, :])
        return output

    @staticmethod
    def _calculate_cnn_output_size(module, input_shape):
        """
        Helper function to calculate the output size of a CNN module.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(*input_shape)
            output = module(dummy_input)
            return output.numel() // output.size(0)
