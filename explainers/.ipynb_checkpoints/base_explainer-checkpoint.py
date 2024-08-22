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