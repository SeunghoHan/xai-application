import numpy as np
import torch

from .lime import LimeExplainer
from .lime_multiterm import MultiTermLIMEExplainer
from .shap import ShapExplainer
from .attention import AttentionExplainer
from .grad_cam import GradCAMExplainer
from .lrp import LRPExplainer
 


# def get_explainer(explainer_type, model, device, train_sequences, 
#                   sequence_length, input_size, selected_features, scaler, 
#                   term="single", long_sequences=None, short_sequences=None, 
#                   long_sequence_length=None, short_sequence_length=None):
#     """
#     Factory function to create an explainer instance.
    
#     Args:
#         explainer_type: Type of explainer ('LIME', 'SHAP', 'Attention', etc.)
#         model: The trained model to explain.
#         device: The device to run the model (e.g., 'cpu' or 'cuda').
#         train_sequences: Training data sequences for single-term.
#         sequence_length: Sequence length for single-term input data.
#         input_size: Number of features in the input data.
#         selected_features: List of feature names.
#         scaler: Scaler used to normalize the data.
#         term: Type of term to explain ('single', 'long', 'short', 'multi').
#         long_sequences: Training sequences for Long-term data.
#         short_sequences: Training sequences for Short-term data.
#         long_sequence_length: Sequence length for Long-term data.
#         short_sequence_length: Sequence length for Short-term data.

#     Returns:
#         An explainer instance.
#     """
#     explainer = None

#     if explainer_type == 'LIME':
#         if term == "multi":
#             # Multi-term LIMEExplainer
#             if long_sequences is None or short_sequences is None or \
#                long_sequence_length is None or short_sequence_length is None:
#                 raise ValueError("Multi-term explanation requires long/short sequences and lengths.")
            
#             explainer = MultiTermLIMEExplainer(
#                 model=model,
#                 device=device,
#                 long_sequences=long_sequences,
#                 short_sequences=short_sequences,
#                 long_sequence_length=long_sequence_length,
#                 short_sequence_length=short_sequence_length,
#                 input_size=input_size,
#                 selected_features=selected_features,
#                 scaler=scaler
#             )
#         else:
#             # Single-term LIMEExplainer
#             explainer = LimeExplainer(
#                 model=model,
#                 device=device,
#                 sequences=train_sequences,
#                 sequence_length=sequence_length,
#                 input_size=input_size,
#                 selected_features=selected_features,
#                 scaler=scaler,
#                 term=term
#             )
#     elif explainer_type == 'SHAP':
#         explainer = ShapExplainer(
#             model=model,
#             sequences=train_sequences,
#             sequence_length=sequence_length,
#             input_size=input_size,
#             selected_features=selected_features,
#             device=device
#         )
#     elif explainer_type == 'Attention':
#         if 'Att' in model.__class__.__name__:
#             explainer = AttentionExplainer(
#                 model=model,
#                 device=device,
#                 sequences=train_sequences,
#                 sequence_length=sequence_length,
#                 input_size=input_size,
#                 selected_features=selected_features
#             )
#         else:
#             print(f"The model does not contain an attention layer. '{explainer_type}' is not applicable.")
#     elif explainer_type == 'GRAD_CAM':
#         explainer = GradCAMExplainer(
#             model=model,
#             device=device,
#             sequences=train_sequences,
#             sequence_length=sequence_length,
#             input_size=input_size,
#             selected_features=selected_features
#         )
#     else:
#         print(f"'{explainer_type}' is not supported.")
    
#     return explainer


def get_explainer(explainer_type, model, device, train_sequences, sequence_length, input_size, selected_features, scaler=None):
    """
    Initialize the appropriate explainer based on the type.

    Parameters:
        explainer_type (str): Type of explainer ('LIME', 'SHAP', 'ATTENTION').
        model (torch.nn.Module): The trained model.
        device (torch.device): Device (CPU or GPU).
        train_sequences (np.array): Training data for the explainer.
        sequence_length (int): Sequence length for the model.
        input_size (int): Input feature size.
        selected_features (list): List of selected features for explanation.
        scaler (object): Scaler for data normalization (if applicable).

    Returns:
        BaseExplainer: The initialized explainer object.
    """
    # Ensure train_sequences is in NumPy format
    if isinstance(train_sequences, torch.Tensor):
        train_sequences = train_sequences.cpu().numpy()  # Convert to NumPy

    
    if explainer_type == 'LIME':
        return LimeExplainer(model, device, train_sequences, sequence_length, input_size, selected_features, scaler)
    elif explainer_type == 'SHAP':
        return ShapExplainer(model, device, train_sequences, sequence_length, input_size, selected_features, scaler)
    elif explainer_type == 'ATTENTION':
        return AttentionExplainer(model, device, selected_features)
    else:
        print(f"Invalid explainer type: {explainer_type}")
        return None


# def get_explainer(explainer_type, model, device, train_sequences, 
#                   sequence_length, input_size, selected_features, scaler):
#     explainer = None
#     if explainer_type == 'LIME':
#         explainer = LimeExplainer(model, device, train_sequences, 
#                                   sequence_length=sequence_length, 
#                                   input_size=input_size,
#                                   selected_features=selected_features, 
#                                   scaler=scaler)
#     elif explainer_type == 'SHAP':
#         explainer = ShapExplainer(model, train_sequences, sequence_length, 
#                               input_size, selected_features, device)
    
#     elif explainer_type == 'Attention':
#         if 'Att' in model.__class__.__name__:
#             explainer = AttentionExplainer(model, device, train_sequences, 
#                                            sequence_length=sequence_length, input_size=input_size,
#                                            selected_features=selected_features)
#         else:
#             print('The model does not contain an attention layer.'.format(explainer_type))

#     elif explainer_type == 'GRAD_CAM':
#         explainer = GradCAMExplainer(model, device, train_sequences, 
#                                      sequence_length=sequence_length, input_size=input_size,
#                                      selected_features=selected_features)
    
#     else:
#         print('\'{}\' is not supported.'.format(explainer_type))

#     return explainer