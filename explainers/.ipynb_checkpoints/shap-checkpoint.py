import shap
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class PyTorchLSTMModelWrapper:
    def __init__(self, model, sequence_length, input_size, device):
        self.model = model
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.device = device

    def __call__(self, x):
        x = x.reshape((x.shape[0], self.sequence_length, self.input_size))
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy()
            
    def predict(self, x):
        x = x.reshape((x.shape[0], self.sequence_length, self.input_size))
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(x).cpu().numpy().flatten()

class ShapExplainer:
    def __init__(self, model, train_sequences, sequence_length, input_size, selected_features, device):
        self.model = model
        self.train_sequences = train_sequences
        self.sequence_length = sequence_length
        self.input_size = input_size
        self.selected_features = selected_features
        self.device = device
        self.explainer = shap.KernelExplainer(PyTorchLSTMModelWrapper(model, sequence_length, input_size, device), train_sequences[:100].reshape(100, -1))
    
    def explain(self, eval_sequences, num_samples=1, nsamples=500):
        random_samples = eval_sequences[np.random.choice(len(eval_sequences), num_samples, replace=False)].reshape(num_samples, -1)
        shap_values = self.explainer.shap_values(random_samples, nsamples=nsamples)
        return shap_values, random_samples

    def sequence_to_dataframe(self, sequence, feature_names):
        reshaped_sequence = sequence.reshape(self.sequence_length, self.input_size)
        return pd.DataFrame(reshaped_sequence, columns=feature_names[:self.input_size])

    def plot_summary(self, shap_values, sample_df, feature_names, sample_index):
        shap.summary_plot(shap_values.reshape(self.sequence_length, -1), sample_df, feature_names=feature_names[:self.input_size * self.sequence_length], show=True)

    def plot_force(self, shap_values, sample_df, feature_names, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=np.repeat(self.explainer.expected_value, self.sequence_length * self.input_size), 
            data=sample_df.values.flatten(), 
            feature_names=feature_names[:self.sequence_length * self.input_size]
        )
        shap.force_plot(
            base_value=self.explainer.expected_value, 
            shap_values=shap_values_explanation.values, 
            features=shap_values_explanation.data, 
            feature_names=shap_values_explanation.feature_names,
            matplotlib=True
        )

    def plot_waterfall(self, shap_values, sample_df, feature_names, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values.flatten(), 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.waterfall_plot(shap_values_explanation)

    def plot_dependence(self, shap_values, sample_df, sample_index):
        for feature in self.selected_features:
            feature_name = f"{feature}_0"
            shap.dependence_plot(feature_name, shap_values.reshape(self.sequence_length, -1), sample_df)

    def plot_decision(self, shap_values, sample_df, feature_names, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values.flatten(), 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.decision_plot(self.explainer.expected_value, shap_values_explanation.values, feature_names=feature_names[:self.input_size * self.sequence_length])

    def plot_scatter(self, shap_values, sample_df, feature_names, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=np.repeat(self.explainer.expected_value, self.sequence_length), 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size]
        )
        for feature in self.selected_features:
            feature_name = f"{feature}_0"
            feature_index = feature_names.index(feature_name)
            shap.plots.scatter(shap_values_explanation[:, feature_index])

    def plot_bar(self, shap_values, sample_df, feature_names, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.plots.bar(shap_values_explanation)

    def plot_beeswarm(self, shap_values, sample_df, feature_names, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.plots.beeswarm(shap_values_explanation)

    def plot_heatmap(self, shap_values, sample_df, feature_names, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.reshape(self.sequence_length, -1), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values, 
            feature_names=feature_names[:self.input_size * self.sequence_length]
        )
        shap.plots.heatmap(shap_values_explanation)

    def plot_partial_dependence(self, shap_values, sample_df, feature_name, sample_index):
        shap_values_explanation = shap.Explanation(
            values=shap_values.flatten(), 
            base_values=self.explainer.expected_value, 
            data=sample_df.values.flatten(), 
            feature_names=[feature_name]
        )
        if feature_name in sample_df.columns:
            shap.dependence_plot(feature_name, shap_values.reshape(self.sequence_length, self.input_size), sample_df)
        else:
            print(f"Feature {feature_name} not found in sample_df columns: {sample_df.columns}")

    def visualize(self, shap_values, eval_sequences):
        shap.initjs()
        feature_names = [f"{feature}_{i}" for i in range(self.sequence_length) for feature in self.selected_features]

        for i in range(len(eval_sequences)):
            sample_df = self.sequence_to_dataframe(eval_sequences[i], feature_names)
            self.plot_summary(shap_values[i], sample_df, feature_names, i)
            self.plot_force(shap_values[i], sample_df, feature_names, i)
            self.plot_waterfall(shap_values[i], sample_df, feature_names, i)
            self.plot_dependence(shap_values[i], sample_df, i)
            self.plot_decision(shap_values[i], sample_df, feature_names, i)
            self.plot_scatter(shap_values[i], sample_df, feature_names, i)
            self.plot_bar(shap_values[i], sample_df, feature_names, i)
            self.plot_beeswarm(shap_values[i], sample_df, feature_names, i)
            self.plot_heatmap(shap_values[i], sample_df, feature_names, i)
            self.plot_partial_dependence(shap_values[i], sample_df, self.selected_features[0], i)
