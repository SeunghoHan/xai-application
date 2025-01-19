import torch
import numpy as np
import matplotlib.pyplot as plt


class LRPExplainer:
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features, epsilon=1e-6):
        self.model = model
        self.device = device
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features
        self.epsilon = epsilon  # 작은 값으로 안정성을 위한 epsilon

    def explain(self, data_point, target_index=None):
        """
        LRP를 통해 입력 데이터 포인트에 대한 relevance 값을 계산합니다.
        """
        self.model.train()  # 모델을 train 모드로 전환하여 backward 계산 가능하도록 설정
        data_tensor = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0).to(self.device)  # 데이터 포인트를 텐서로 변환
    
        # requires_grad=True로 설정하여 gradient 계산 가능하게
        data_tensor.requires_grad = True
    
        # Forward pass
        output = self.model(data_tensor)
        if target_index is not None:
            output = output[:, target_index]
    
        # Backward pass to get gradients
        self.model.zero_grad()
        output.backward(torch.ones_like(output))
    
        # relevance: start from the output gradient
        relevance = data_tensor.grad.cpu().numpy().squeeze()
    
        # LRP 계산
        relevance_values = self.lrp(relevance, data_tensor.cpu().detach().numpy())
    
        self.model.eval()  # 모델을 다시 eval 모드로 전환
    
        return relevance_values
    
    def lrp(self, relevance, activations):
        """
        LRP-ε 규칙을 사용하여 relevance를 입력으로 역전파합니다.
        """
        activations = activations.squeeze()
        relevance_values = np.zeros_like(activations)
    
        if activations.ndim == 2:
            # 2차원 배열 처리 (sequence_length, input_size)
            for i in range(self.input_size):
                z = activations[:, i] + self.epsilon * np.where(activations[:, i] >= 0, 1, -1)
                s = relevance[:, np.newaxis] / (z.sum(axis=1, keepdims=True) + self.epsilon)  # 각 샘플에 맞춰 relevance 계산
                relevance_values[:, i] = activations[:, i] * s.squeeze()
        elif activations.ndim == 1:
            # 1차원 배열 처리
            z = activations + self.epsilon * np.where(activations >= 0, 1, -1)
            s = relevance / (z.sum() + self.epsilon)
            relevance_values = activations * s
    
        return relevance_values
        
    def visualize_lrp(self, relevance_values, data_point):
        """
        LRP 결과를 시각화합니다.
        """
        sequence_length, input_size = data_point.shape
        relevance_sequence_length = relevance_values.shape[0]

        if sequence_length != relevance_sequence_length:
            raise ValueError(f"Data point sequence length ({sequence_length}) and relevance values length ({relevance_sequence_length}) do not match")

        feature_colors = ['blue', 'green', 'orange', 'purple']

        plt.figure(figsize=(10, 6))

        for i in range(input_size):
            plt.plot(relevance_values[:, i], label=f'{self.selected_features[i]} LRP', color=feature_colors[i % len(feature_colors)])
            plt.fill_between(range(sequence_length), relevance_values[:, i], color=feature_colors[i % len(feature_colors)], alpha=0.3)

        plt.title("LRP Feature Contribution")
        plt.xlabel("Time Step")
        plt.ylabel("LRP Contribution")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

