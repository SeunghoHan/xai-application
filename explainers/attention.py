import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from .base_explainer import BaseExplainer

class AttentionExplainer(BaseExplainer):
    def __init__(self, model, device, sequences, sequence_length, input_size, selected_features):
        # BaseExplainer의 초기화
        super().__init__(model, device, sequences, sequence_length, input_size, selected_features)
    
    def explain(self, data_point):
        """Explain a specific data point by analyzing attention weights."""
        self.model.eval()
        data_tensor = torch.tensor(data_point, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 모델 예측 수행 (Attention과 함께 반환되는 구조)
        output, attention_weights = self.model(data_tensor)
        attention_weights = attention_weights.squeeze().cpu().detach().numpy()
        
        return attention_weights, output.item()
    
    def visualize_attention(self, data_point):
        """Visualize the attention weights along with input features for a specific data point."""
        attention_weights, prediction = self.explain(data_point)
        
        # 시각화 설정
        plt.figure(figsize=(10, 6))

        # 입력된 시퀀스의 feature 시각화
        sequence_np = np.array(data_point)  # data_point는 입력 시퀀스 (sequence_length, input_size)
        feature_colors = ['blue', 'green', 'orange', 'purple']  # 각 feature에 대한 색상 지정
        feature_names = self.selected_features  # Feature 이름
        
        # Feature 값 그래프 그리기
        for i, feature_name in enumerate(feature_names):
            plt.plot(sequence_np[:, i], label=feature_name, color=feature_colors[i])
        
        # Attention 가중치 시각화 (다른 축에 시각화)
        ax2 = plt.gca().twinx()  # 두 번째 축 (Y축)을 추가
        ax2.plot(attention_weights, label='Attention Weights', color='red', linestyle='--', alpha=0.6)
        ax2.fill_between(range(len(attention_weights)), attention_weights, color='red', alpha=0.3)

        # 그래프 설정
        plt.title("Input Features and Attention Weights for a Data Point")
        plt.xlabel("Time Step")
        plt.ylabel("Feature Values")
        ax2.set_ylabel("Attention Weights")
        
        # 레전드 외부에 추가
        plt.figlegend([plt.Line2D([0], [0], color=feature_colors[i]) for i in range(len(feature_names))] + 
                      [plt.Line2D([0], [0], color='red', linestyle='--')],
                      feature_names + ['Attention Weights'],
                      loc='upper center', ncol=len(feature_names) + 1, bbox_to_anchor=(0.5, -0.1))

        plt.tight_layout()
        plt.show()
        
        print(f"Prediction for the data point: {prediction}")
    
    def analyze_attention_distribution(self, num_samples=300):
        """Analyze the attention weight distribution across multiple samples."""
        attention_weights_list = []
        
        # 샘플 몇 개를 선택하여 분석
        indices = np.random.choice(len(self.sequences), num_samples, replace=False)
        selected_sequences = self.sequences[indices]
        
        for sequence in selected_sequences:
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            _, attention_weights = self.model(sequence_tensor)
            attention_weights = attention_weights.squeeze().cpu().detach().numpy()
            attention_weights_list.append(attention_weights)
        
        # 평균과 표준편차 계산
        attention_weights_arr = np.array(attention_weights_list)
        mean_attention = np.mean(attention_weights_arr, axis=0)
        std_attention = np.std(attention_weights_arr, axis=0)
        
        # 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(mean_attention, label='Mean Attention Weights', color='red')
        plt.fill_between(range(len(mean_attention)), mean_attention - std_attention, mean_attention + std_attention, color='red', alpha=0.3)
        plt.title(f"Mean and Standard Deviation of Attention Weights across {num_samples} Samples")
        plt.xlabel("Time Step")
        plt.ylabel("Attention Weight")
        plt.legend()
        plt.show()

    def visualize_attention_heatmap(self, data_point):
        """Attention Heatmap Visualization."""
        attention_weights, prediction = self.explain(data_point)

        # Attention 가중치를 Heatmap으로 시각화
        plt.figure(figsize=(10, 6))
        sns.heatmap(attention_weights[np.newaxis, :], cmap="viridis", cbar=True, xticklabels=range(len(attention_weights)), yticklabels=['Attention'])
        plt.title("Attention Weights Heatmap for a Data Point")
        plt.xlabel("Time Step")
        plt.ylabel("Attention")
        plt.tight_layout()
        plt.show()

        print(f"Prediction for the data point: {prediction}")
    
    def visualize_cumulative_attention(self, data_point, num_points=7):
        """
        Cumulative Attention Weights Visualization with important points.
        
        Args:
            data_point: 입력 데이터 포인트
            num_points: 표시할 기울기 변화가 큰 포인트의 개수
        """
        attention_weights, prediction = self.explain(data_point)

        # Attention 가중치의 누적 합 계산
        cumulative_attention = np.cumsum(attention_weights)

        # 기울기의 변화를 계산 (차분)
        gradient = np.diff(cumulative_attention)

        # 기울기 변화가 큰 상위 num_points 지점 찾기
        top_indices = np.argsort(np.abs(gradient))[-num_points:]  # 기울기 변화가 큰 num_points 곳을 찾음
        
        # 누적 Attention 시각화
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_attention, label='Cumulative Attention', color='blue')

        # 중요한 기울기 변화 지점에 점과 라벨 추가 ('top {x} in step {x}' 형식)
        for idx, step in enumerate(sorted(top_indices), 1):
            plt.scatter(step, cumulative_attention[step], color='red', zorder=5)
            plt.annotate(f"Top {idx} in step {step}", (step, cumulative_attention[step]), 
                         textcoords="offset points", xytext=(0, 15), ha='center', color='red')

        # 그래프 설정
        plt.title(f"Cumulative Attention Over Time with Top {num_points} Gradient Changes")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Attention")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"Prediction for the data point: {prediction}")
        print(f"Top {num_points} steps with largest gradient changes: {sorted(top_indices)}")