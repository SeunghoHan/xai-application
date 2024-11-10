import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

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
        
        return attention_weights, output # output.item()
    
    def visualize_attention(self, data_point=None, n=5):
        """
        Visualize the attention weights along with input features for a specific data point
        and print the feature values at the top n attention-weighted time steps. The top n
        time steps are also highlighted on the plot with a label.
        
        Args:
            data_point: 입력 데이터 포인트 (sequence_length, input_size)
            n: attention 가중치가 가장 큰 상위 n개의 타임스텝에 대해 출력할 feature 값 수
        """
        if data_point == None:
            data_point = self.sequences[np.random.choice(len(self.sequences))]
        
        attention_weights, prediction = self.explain(data_point)
        
        # 시각화 설정
        plt.figure(figsize=(10, 6))

        # 입력된 시퀀스의 feature 시각화
        sequence_np = np.array(data_point)  # data_point는 입력 시퀀스 (sequence_length, input_size)
        feature_colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'black']  # 각 feature에 대한 색상 지정
        feature_names = self.selected_features  # Feature 이름
        feature_colors = feature_colors[:len(feature_names)]
        
        # Feature 값 그래프 그리기
        for i, feature_name in enumerate(feature_names):
            plt.plot(sequence_np[:, i], label=feature_name, color=feature_colors[i])
        
        # Attention 가중치 시각화 (다른 축에 시각화)
        ax2 = plt.gca().twinx()  # 두 번째 축 (Y축)을 추가
        ax2.plot(attention_weights, label='Attention Weights', color='red', linestyle='--', alpha=0.6)
        ax2.fill_between(range(len(attention_weights)), attention_weights, color='red', alpha=0.3)

        # 상위 n개의 타임스텝 추출
        top_n_indices = np.argsort(attention_weights)[-n:][::-1]  # 상위 n개 인덱스를 attention 크기 순으로 정렬
        
        # 그래프에 top n 위치 표시
        for idx, step in enumerate(top_n_indices, 1):
            ax2.scatter(step, attention_weights[step], color='red', zorder=5)
            ax2.annotate(f"Top {idx}", (step, attention_weights[step]), 
                         textcoords="offset points", xytext=(0, 10), ha='center', color='red')

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
        
        # Attention 가중치가 큰 상위 n개 타임스텝을 출력
        print(f"Prediction for the data point: {prediction}")
        print(f"Top {n} time steps with highest attention weights:")
        
        for idx, step in enumerate(top_n_indices, 1):
            print(f"Top {idx} - Time step {step}: Attention Weight = {attention_weights[step]}")
            for i, feature_name in enumerate(feature_names):
                print(f"  {feature_name}: {sequence_np[step, i]}")
                
    
    def analyze_attention_distribution(self, num_samples=300):
        """Analyze the attention weight distribution across multiple samples"""
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

    def visualize_attention_heatmap(self, data_point=None):
        """Attention Heatmap Visualization."""
        if data_point == None:
            data_point = self.sequences[np.random.choice(len(self.sequences))]
            
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
    
    def visualize_cumulative_attention(self, data_point=None, num_points=7):
        """
        Cumulative Attention Weights Visualization with important points.
        
        Args:
            data_point: 입력 데이터 포인트
            num_points: 표시할 기울기 변화가 큰 포인트의 개수
        """
        if data_point == None:
            data_point = self.sequences[np.random.choice(len(self.sequences))]
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
            plt.annotate(f"Top {idx}", (step, cumulative_attention[step]), 
                         textcoords="offset points", xytext=(0, 15), ha='center', color='red')

        # 그래프 설정
        plt.title(f"Cumulative Attention Over Time with Top {num_points} Gradient Changes")
        plt.xlabel("Time Step")
        plt.ylabel("Cumulative Attention")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Top n 포인트에 대한 feature 값을 출력
        sequence_np = np.array(data_point)  # 입력 데이터 포인트 (sequence_length, input_size)
        feature_names = self.selected_features  # 선택된 feature 리스트

        print(f"Prediction for the data point: {prediction}")
        print(f"Top {num_points} steps with largest gradient changes:")

        for idx, step in enumerate(sorted(top_indices), 1):
            print(f"Top {idx} - Time step {step}: Gradient Change = {gradient[step-1]}")  # 기울기 변화값 출력
            for i, feature_name in enumerate(feature_names):
                print(f"  {feature_name}: {sequence_np[step, i]}")  # 해당 타임스텝의 feature 값 출력


    def extract_important_features(self, num_samples=100, top_n=5):
        """
        여러 샘플에 대해 Attention weights를 분석하여, 주요 feature를 추출합니다.
        
        Args:
            num_samples: 분석할 샘플의 수
            top_n: 상위 주요 feature 개수
        
        Returns:
            주요 feature와 그들의 빈도를 포함한 리스트
        """
        # 주요 feature를 집계할 Counter 객체
        feature_counter = Counter()
        
        # num_samples 개의 랜덤 샘플 선택
        indices = np.random.choice(len(self.sequences), num_samples, replace=False)
        selected_sequences = self.sequences[indices]
        
        # 각 샘플에 대해 attention weight 분석
        for sequence in tqdm(selected_sequences, desc="Extracting important features", unit="sample"):
            attention_weights, _ = self.explain(sequence)
            
            # Attention weight가 가장 높은 타임스텝의 feature 선택
            top_indices = np.argsort(attention_weights)[-top_n:]
            for step in top_indices:
                feature_importance = sequence[step] * attention_weights[step]  # 각 feature의 실제 값과 가중치 곱
                for i, feature in enumerate(self.selected_features):
                    feature_counter[feature] += abs(feature_importance[i])
        
        # 상위 주요 feature 추출 및 딕셔너리 변환
        top_features = dict(feature_counter.most_common(top_n))
        
        # 결과 출력
        print(f"\nTop {top_n} important features:")
        for i, (feature, importance) in enumerate(top_features.items(), 1):
            print(f"Top {i}: {feature} (Importance: {importance:.4f})")
    
        return top_features