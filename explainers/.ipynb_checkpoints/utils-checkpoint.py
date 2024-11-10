import numpy as np

from .lime import LimeExplainer
from .shap import ShapExplainer
from .attention import AttentionExplainer
from .grad_cam import GradCAMExplainer
from .lrp import LRPExplainer

def get_explainer(explainer_type, model, device, train_sequences, 
                  sequence_length, input_size, selected_features, scaler):
    explainer = None
    if explainer_type == 'LIME':
        explainer = LimeExplainer(model, device, train_sequences, 
                                  sequence_length=sequence_length, 
                                  input_size=input_size,
                                  selected_features=selected_features, 
                                  scaler=scaler)
    elif explainer_type == 'SHAP':
        explainer = ShapExplainer(model, train_sequences, sequence_length, 
                              input_size, selected_features, device)
    
    elif explainer_type == 'Attention':
        if 'Att' in model.__class__.__name__:
            explainer = AttentionExplainer(model, device, train_sequences, 
                                           sequence_length=sequence_length, input_size=input_size,
                                           selected_features=selected_features)
        else:
            print('The model does not contain an attention layer.'.format(explainer_type))

    elif explainer_type == 'GRAD_CAM':
        explainer = GradCAMExplainer(model, device, train_sequences, 
                                     sequence_length=sequence_length, input_size=input_size,
                                     selected_features=selected_features)
    
    else:
        print('\'{}\' is not supported.'.format(explainer_type))

    return explainer