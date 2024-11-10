import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib.pyplot as plt


from .base_explainer import BaseExplainer

class GradCAMExplainer(BaseExplainer):
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features):
        """
        Grad-CAM을 사용하여 모델의 타임스텝 기여도를 계산하는 클래스
        
        Args:
            model: 학습된 LSTM 기반 모델
            device: 모델이 할당된 장치 (cpu 또는 cuda)
            sequences: 사용된 데이터 시퀀스 (train 또는 eval data)
            sequence_length: 시퀀스 길이
            input_size: feature 개수
            selected_features: 사용된 feature의 이름 리스트
        """
        super().__init__(model, device, sequences, sequence_length, input_size, selected_features)


    def compute_grad_cam(self, data_point, target_index=None):
        """
        Grad-CAM을 학습 모델에 적용하여 각 타임스텝의 기여도를 계산
        
        Args:
            data_point: 시계열 데이터 포인트 (sequence_length, input_size)
            target_index: 목표 예측값의 인덱스 (분류 모델에서 사용 가능)
        
        Returns:
            Grad-CAM으로 계산된 각 타임스텝의 기여도 맵
        """
        self.model.train()  # 모델을 train 모드로 전환하여 backward 계산 가능하도록 설정

        data_tensor = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, sequence_length, input_size)
        data_tensor.requires_grad_(True)

        gradients = []
        
        def hook_fn(grad):
            gradients.append(grad)
        
        output, _ = self.model(data_tensor)
        if target_index is not None:
            output = output[:, target_index]
        
        self.model.zero_grad()
        
        handle = data_tensor.register_hook(hook_fn)
        output.backward(torch.ones_like(output))
        handle.remove()

        gradients = gradients[0].cpu().detach().numpy().squeeze()  # (sequence_length, input_size)

        grad_cam_weights = np.mean(gradients, axis=1)  # (sequence_length,)
        
        self.model.eval()  # 모델을 다시 eval 모드로 전환
        
        return grad_cam_weights

    def compute_grad_cam_per_feature(self, data_point, target_index=None):
        """
        Grad-CAM을 학습 모델에 적용하여 각 타임스텝과 각 feature별 기여도를 계산
        
        Args:
            data_point: 시계열 데이터 포인트 (sequence_length, input_size)
            target_index: 목표 예측값의 인덱스 (분류 모델에서 사용 가능)
        
        Returns:
            Grad-CAM으로 계산된 각 타임스텝과 각 feature의 기여도 맵 (sequence_length, input_size)
        """
        self.model.train()  # 모델을 train 모드로 전환하여 backward 계산 가능하도록 설정
        
        data_tensor = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, sequence_length, input_size)
        data_tensor.requires_grad_(True)

        gradients = []
        
        def hook_fn(grad):
            gradients.append(grad)
        
        output, _ = self.model(data_tensor)
        if target_index is not None:
            output = output[:, target_index]
        
        self.model.zero_grad()
        
        handle = data_tensor.register_hook(hook_fn)
        output.backward(torch.ones_like(output))
        handle.remove()

        gradients = gradients[0].cpu().detach().numpy().squeeze()  # (sequence_length, input_size)
        
        self.model.eval()  # 모델을 다시 eval 모드로 전환
        
        return gradients  # (sequence_length, input_size)

    def visualize_grad_cam(self, grad_cam_weights, data_point, selected_features):
        """
        Grad-CAM으로 계산된 타임스텝 기여도를 시각화
        
        Args:
            grad_cam_weights: Grad-CAM으로 계산된 가중치 (sequence_length,)
            data_point: 시계열 데이터 포인트 (sequence_length, input_size)
            selected_features: 사용된 feature의 이름 리스트
        """
        sequence_length = data_point.shape[0]
        
        plt.figure(figsize=(16, 8))
        plt.plot(grad_cam_weights, label='Grad-CAM Weights', color='red', linestyle='--')
        plt.fill_between(range(sequence_length), grad_cam_weights, color='red', alpha=0.3)
        
        feature_colors = ['blue', 'green', 'orange', 'purple']
        for i in range(data_point.shape[1]):
            plt.plot(data_point[:, i], label=selected_features[i], color=feature_colors[i % len(feature_colors)])
        
        plt.title("LSTM Grad-CAM Visualization")
        plt.xlabel("Time Step")
        plt.ylabel("Contribution")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(selected_features) + 1)
        plt.tight_layout()
        plt.show()

    def visualize_grad_cam_per_feature(self, grad_cam_weights, data_point, selected_features):
        """
        Grad-CAM으로 계산된 타임스텝과 feature별 기여도를 시각화
        
        Args:
            grad_cam_weights: Grad-CAM으로 계산된 가중치 (sequence_length, input_size)
            data_point: 시계열 데이터 포인트 (sequence_length, input_size)
            selected_features: 사용된 feature의 이름 리스트
        """
        sequence_length, input_size = grad_cam_weights.shape
        
        feature_colors = ['blue', 'green', 'orange', 'purple']
        
        plt.figure(figsize=(16, 8))
        for i in range(input_size):
            plt.plot(grad_cam_weights[:, i], label=f'{selected_features[i]} Grad-CAM', color=feature_colors[i % len(feature_colors)])
            plt.fill_between(range(sequence_length), grad_cam_weights[:, i], color=feature_colors[i % len(feature_colors)], alpha=0.3)
        
        plt.title("Combined Grad-CAM Feature Contribution")
        plt.xlabel("Time Step")
        plt.ylabel("Feature Contribution")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(selected_features))
        plt.tight_layout()
        plt.show()

        for i in range(input_size):
            plt.figure(figsize=(12, 7))
            plt.plot(grad_cam_weights[:, i], label=f'{selected_features[i]} Grad-CAM', color=feature_colors[i % len(feature_colors)])
            plt.fill_between(range(sequence_length), grad_cam_weights[:, i], color=feature_colors[i % len(feature_colors)], alpha=0.3)
            
            plt.title(f"{selected_features[i]} Grad-CAM Feature Contribution")
            plt.xlabel("Time Step")
            plt.ylabel("Feature Contribution")
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()


    def compute_grad_cam_per_feature(self, data_point, target_index=None):
        """
        각 타임스텝과 feature별 기여도를 계산하는 Grad-CAM
        
        Args:
            data_point: 시계열 데이터 포인트 (sequence_length, input_size)
            target_index: 목표 예측값의 인덱스 (분류 모델에서 사용 가능)
        
        Returns:
            Grad-CAM으로 계산된 각 타임스텝과 각 feature의 기여도 맵 (sequence_length, input_size)
        """
        self.model.train()
        
        data_tensor = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0).to(self.device)
        data_tensor.requires_grad_(True)

        gradients = []
        
        def hook_fn(grad):
            gradients.append(grad)
        
        output, _ = self.model(data_tensor)
        if target_index is not None:
            output = output[:, target_index]
        
        self.model.zero_grad()
        
        handle = data_tensor.register_hook(hook_fn)
        output.backward(torch.ones_like(output))
        handle.remove()

        gradients = gradients[0].cpu().detach().numpy().squeeze()
        
        self.model.eval()
        
        return gradients  # (sequence_length, input_size)

    def extract_important_features(self, num_samples=50, top_n=3):
        """
        여러 샘플에 대해 Grad-CAM을 수행하고, 중요도가 큰 상위 N개의 feature를 추출합니다.
        
        Args:
            num_samples: Grad-CAM을 적용할 샘플 수
            top_n: 중요도가 높은 상위 feature 수
        
        Returns:
            모델 예측에 많은 영향을 미치는 상위 feature 리스트
        """
        feature_importance = {feature: 0 for feature in self.selected_features}

        indices = np.random.choice(len(self.sequences), num_samples, replace=False)
        selected_sequences = self.sequences[indices]

        print("Extracting important features with Grad-CAM...")
        
        for i, data_point in enumerate(tqdm(selected_sequences, desc="Processing samples"), start=1):
            grad_cam_weights = self.compute_grad_cam_per_feature(data_point)
            
            # 타임스텝과 feature별로 평균 기여도를 계산하여 feature 중요도에 반영
            avg_importance_per_feature = np.mean(np.abs(grad_cam_weights), axis=0)
            
            for feature_idx, feature_name in enumerate(self.selected_features):
                feature_importance[feature_name] += avg_importance_per_feature[feature_idx]

        # 중요도가 큰 상위 N개 feature 추출
        top_features = sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)[:top_n]
        top_features_dict = {feature: importance for feature, importance in top_features}
        
        print(f"Top {top_n} important features based on SHAP values:")
        for rank, (feature, importance) in enumerate(top_features, start=1):
            print(f"Top {rank}: {feature} (Importance: {importance})")
        
        return top_features_dict
