import os
import re
import json
import numpy as np
import torch
import lime
from tqdm import tqdm
from lime import lime_tabular
import matplotlib.pyplot as plt

from .base_explainer import BaseExplainer

class MultiTermLIMEExplainer(BaseExplainer):
    def __init__(self, model, device, long_sequences, short_sequences, 
                 long_sequence_length, short_sequence_length, 
                 input_size, selected_features, scaler):
        """
        MultiTerm LIME Explainer for handling both Long-term and Short-term sequences.
        
        Args:
            model: The multi-term model to explain.
            device: The device to run the model (e.g., 'cuda' or 'cpu').
            long_sequences: Training sequences for Long-term data.
            short_sequences: Training sequences for Short-term data.
            long_sequence_length: Sequence length for Long-term data.
            short_sequence_length: Sequence length for Short-term data.
            input_size: Number of features in the input data.
            selected_features: List of feature names.
            scaler: Scaler used to normalize/denormalize the data.
        """
        super().__init__(model, device, None, None, input_size, selected_features)
        self.long_sequences = long_sequences
        self.short_sequences = short_sequences
        self.long_sequence_length = long_sequence_length
        self.short_sequence_length = short_sequence_length
        self.scaler = scaler
        self.long_explainer = self.create_explainer(self.long_sequences, "long")
        self.short_explainer = self.create_explainer(self.short_sequences, "short")

    def create_explainer(self, sequences, term):
        """
        Create LimeTabularExplainer for Long-term or Short-term sequences.
        """
        sequence_length = self.long_sequence_length if term == "long" else self.short_sequence_length
        return lime_tabular.LimeTabularExplainer(
            training_data=sequences.reshape(-1, sequence_length * self.input_size),
            feature_names=[f"{term}_{feature}_{i}" for i in range(sequence_length) for feature in self.selected_features],
            class_names=['Global_active_power'],
            mode='regression'
        )

    def predict_fn(self, input_data):
        """
        Predict function for LIME.
        Combines Long-term and Short-term inputs for Multi-term models.
        """
        long_input = input_data["long"]
        short_input = input_data["short"]
    
        # Ensure inputs are PyTorch tensors
        long_input_tensor = torch.tensor(long_input, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add channel dimension for CNN
        short_input_tensor = torch.tensor(short_input, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add channel dimension for CNN
    
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(long_input=long_input_tensor, short_input=short_input_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        return outputs.cpu().numpy()

    def explain(self, data_point, eval_long, eval_short, num_features=10, is_visual=False):
        """
        Explain the model's prediction for a given data point.
        """
        # Split Long-term and Short-term inputs
        long_data_point = data_point["long"].reshape(-1)
        short_data_point = data_point["short"].reshape(-1)

        # Long-term explanation
        long_explanation = self.long_explainer.explain_instance(
            long_data_point, lambda x: self.predict_fn({"long": x, "short": eval_short}), 
            num_samples=1000, num_features=num_features
        )

        # Short-term explanation
        short_explanation = self.short_explainer.explain_instance(
            short_data_point, lambda x: self.predict_fn({"long": eval_long, "short": x}), 
            num_samples=1000, num_features=num_features
        )

        # Visualization (Optional)
        if is_visual:
            print("\nLong-term Explanation:")
            long_explanation.show_in_notebook(show_table=True)
            print("\nShort-term Explanation:")
            short_explanation.show_in_notebook(show_table=True)

        return {
            "long": long_explanation.as_list(),
            "short": short_explanation.as_list()
        }

    def extract_important_features(self, eval_long, eval_short, num_samples=50, top_n=3):
        """
        Extract top N important features for Long-term and Short-term inputs.
        """
        feature_importance = {"long": {}, "short": {}}

        # Initialize feature dictionaries
        for term in ["long", "short"]:
            feature_importance[term] = {feature: 0 for feature in self.selected_features}

        indices = np.random.choice(len(eval_long), num_samples, replace=False)

        for idx in tqdm(indices, desc="Processing samples for Multi-term explanation"):
            data_point = {"long": eval_long[idx], "short": eval_short[idx]}

            explanations = self.explain(data_point, eval_long, eval_short, num_features=len(self.selected_features))

            for term in ["long", "short"]:
                for feature_description, importance in explanations[term]:
                    actual_feature = feature_description.split("_")[0]
                    feature_importance[term][actual_feature] += abs(importance)

        # Sort and select top N features
        top_features = {
            term: sorted(features.items(), key=lambda x: -x[1])[:top_n]
            for term, features in feature_importance.items()
        }

        # Format results
        return {
            term: {feature: score for feature, score in top_features[term]}
            for term in ["long", "short"]
        }