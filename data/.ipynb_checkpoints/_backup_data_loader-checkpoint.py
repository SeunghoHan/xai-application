import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class PowerConsumptionDataset:
    def __init__(self, file_path, feature_idx=[], num_of_features=7, sequence_length=60):
        self.file_path = file_path
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()

        all_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        
        if len(feature_idx)>0:
            self.selected_features = [all_features[i] for i in feature_idx if i < len(all_features)]
        else:
            self.selected_features = all_features
    
        # self.selected_features = all_features[0:num_of_features]
        
    def load_data(self):
        # Load and preprocess data
        data = pd.read_csv(self.file_path, sep=';', parse_dates={'datetime': ['Date', 'Time']}, 
                           infer_datetime_format=True, low_memory=False, na_values=['?'])

        # Fill missing values
        data.fillna(data.mean(), inplace=True)

        # Select all features
        data_selected = data[self.selected_features]

        # Normalize the data
        data_scaled = self.scaler.fit_transform(data_selected.values)

        # Create sequences
        sequences, targets = self.create_sequences(data_scaled)

        # Split the data into train and eval sets
        train_sequences, eval_sequences, train_targets, eval_targets = train_test_split(sequences, targets, test_size=0.2, random_state=42)

        return train_sequences, eval_sequences, train_targets, eval_targets

    def create_sequences(self, data):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            targets.append(data[i + self.sequence_length, 0])  # Predicting 'Global_active_power'
        return np.array(sequences), np.array(targets)