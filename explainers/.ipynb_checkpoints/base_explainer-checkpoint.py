import os
import json
import numpy as np
import torch

class BaseExplainer:
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features):
        self.model = model
        self.device = device
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features

    def explain(self, data_point):
        raise NotImplementedError("The explain method must be implemented by the subclass.")

    def inverse_transform_time_features(self, sequences, selected_features, scaler):
        """
        정규화된 시퀀스 데이터를 원래 값으로 복구하고, 시간 피처(`month`, `day`, `hour`)도 복구
    
        Args:
            sequences (numpy.ndarray): 정규화된 시퀀스 데이터
            selected_features (list): 선택된 피처 목록
            scaler (object): MinMaxScaler 또는 StandardScaler 객체
        
        Returns:
            numpy.ndarray: 복구된 시퀀스 데이터 (실제 값)
        """
        # 전체 시퀀스 데이터의 정규화된 부분을 원래 값으로 복구
        original_sequences = scaler.inverse_transform(sequences.reshape(-1, len(selected_features))).reshape(sequences.shape)
    
        # # 시간 관련 피처를 복구하는 부분
        # for i in range(len(sequences)):
        #     for time_step in range(sequences.shape[1]):
        #         if 'month_sin' in selected_features and 'month_cos' in selected_features:
        #             month_sin = original_sequences[i][time_step, selected_features.index('month_sin')]
        #             month_cos = original_sequences[i][time_step, selected_features.index('month_cos')]
        #             original_sequences[i][time_step, selected_features.index('month_sin')] = (np.arctan2(month_sin, month_cos) / (2 * np.pi)) * 12 % 12
    
        #         if 'day_sin' in selected_features and 'day_cos' in selected_features:
        #             day_sin = original_sequences[i][time_step, selected_features.index('day_sin')]
        #             day_cos = original_sequences[i][time_step, selected_features.index('day_cos')]
        #             original_sequences[i][time_step, selected_features.index('day_sin')] = (np.arctan2(day_sin, day_cos) / (2 * np.pi)) * 31 % 31
    
        #         if 'hour_sin' in selected_features and 'hour_cos' in selected_features:
        #             hour_sin = original_sequences[i][time_step, selected_features.index('hour_sin')]
        #             hour_cos = original_sequences[i][time_step, selected_features.index('hour_cos')]
        #             original_sequences[i][time_step, selected_features.index('hour_sin')] = (np.arctan2(hour_sin, hour_cos) / (2 * np.pi)) * 24 % 24
        
        return original_sequences