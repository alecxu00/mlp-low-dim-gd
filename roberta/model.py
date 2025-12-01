import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset 
from transformers import RobertaConfig, RobertaForSequenceClassification 
from utils import left_right_inits
from typing import List 



'''
    RobertaClassificationHead but with (potentially) more hidden layers 
    Using GELU instead of Tanh (like in HuggingFace implementation)
'''
class CustomRobertaClassificationHead(nn.Module):
    
    def __init__(self, 
                 config: RobertaConfig, 
                 num_layers: int = 2,
                 init_scale: float = 1.0):
        
        super(CustomRobertaClassificationHead, self).__init__()
        self.config = config
        self.num_layers = num_layers
        self.init_scale = init_scale

        # Create layers 
        self.layers = self._create_layers()

        # Initialize layers to be epsilon-scaled orthogonal matrices 
        self._init_weights()
        
    
    def _create_layers(self):
        # Dropout probability
        dropout = self.config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob

        # Create L-layer MLP classifier 
        layers_list = []
        for l in range(self.num_layers):
            # Dropout before every linear layer 
            layers_list.append(nn.Dropout(dropout))

            if l == self.num_layers - 1: # Last-layer classifier
                layers_list.append(nn.Linear(self.config.hidden_size, self.config.num_labels))
            else:
                # Hidden linear layer 
                layers_list.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))

                # Activation function (tanh)
                layers_list.append(nn.GELU())
        

        return nn.ModuleList(layers_list)



    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                # Epsilon-scaled orthogonal initialization 
                nn.init.orthogonal_(m.weight, gain=self.init_scale)


    def get_weights(self):
        weights = []

        # Loop through all layers and store weight matrices as numpy array 
        for m in self.layers:
            if isinstance(m, nn.Linear):
                weights.append(m.weight.detach().cpu().numpy())
        
        return weights 

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])

        for layer in self.layers:
            x = layer(x)
        return x


'''
    RobertaForSequenceClassification but with a potentially deeper or compressed classification head 
'''
class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, 
                 config: RobertaConfig,
                 num_classifier_layers: int = 2,
                 classifier_init_scale: float = 1.0, 
                 **kwargs):
        
        super(CustomRobertaForSequenceClassification, self).__init__(config)

        self.classifier = CustomRobertaClassificationHead(config, 
                                                          num_layers=num_classifier_layers, 
                                                          init_scale=classifier_init_scale
                                                          )

# Helper function to initialize linear layer weights 
def init_weight(in_shape: int, 
                out_shape: int,
                weights: np.ndarray = None,
                init_scale: float = 1.0) -> nn.Linear:
    
    # Create linear layer 
    layer = nn.Linear(in_shape, out_shape)

    # Initialize to specific weights if given
    if weights is not None:
        assert weights.shape == (out_shape, in_shape) # Dimensions flipped 

        with torch.no_grad():
            layer.weight.copy_(torch.Tensor(weights))

    # If no weights given, initialize randomly
    else:
        nn.init.orthogonal_(layer.weight, gain=init_scale)

    return layer 

# Helper function to compress two-layer MLP 
def reparameterize_two_layer_classifier(config: RobertaConfig, 
                                        U_init: np.ndarray = None, 
                                        V_init: np.ndarray = None, 
                                        rank: int = None, 
                                        init_scale: float = 1.0) -> List[nn.Module]:
    
    # Re-parameterize two-layer classifier 
    dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob

    comp_layers_list = []

    
    # ====== HIDDEN LAYER ===== #
    # Dropout
    comp_layers_list.append(nn.Dropout(dropout))

    # Right subspace factor
    weights_V = V_init.T if V_init is not None else None
    comp_layers_list.append(init_weight(config.hidden_size, 
                                        rank, 
                                        weights=weights_V, 
                                        init_scale=1.0))

    # Low-rank weight matrix
    comp_layers_list.append(init_weight(rank, 
                                        rank, 
                                        weights = None,
                                        init_scale=init_scale))

    # Left subspace factor
    comp_layers_list.append(init_weight(rank, 
                                        config.hidden_size, 
                                        weights=U_init, 
                                        init_scale=1.0))


    # Activation function
    comp_layers_list.append(nn.GELU())

    # ====== FINAL LAYER =====
    # Dropout 
    comp_layers_list.append(nn.Dropout(dropout))

    # Final layer classifier 
    comp_layers_list.append(init_weight(
                                        config.hidden_size, 
                                        config.num_labels, 
                                        init_scale=init_scale) 
                            )


    return comp_layers_list

# Helper function to compress deeper MLP 
def reparameterize_deep_classifier(num_layers: int, 
                                   config: RobertaConfig, 
                                   U_init: np.ndarray = None, 
                                   V_init: np.ndarray = None, 
                                   rank: int = None, 
                                   init_scale: float = 1.0) -> List[nn.Module]:

    dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob

    comp_layers_list = []
    for l in range(num_layers):
        # Dropout before every layer 
        comp_layers_list.append(nn.Dropout(dropout))

        # First layer: V factor 
        if l == 0:
            # Right subspace factor 
            comp_layers_list.append(init_weight(config.hidden_size, 
                                                rank, 
                                                weights = V_init, 
                                                init_scale = 1.0))

            # Low-rank weight matrix
            comp_layers_list.append(init_weight(rank, 
                                                rank, 
                                                weights = None, 
                                                init_scale = init_scale))

            # Activation function: tanh
            comp_layers_list.append(nn.GELU())
        
        # Penultimate layer: U factor
        elif l == num_layers - 2:
            # Low-rank weight
            comp_layers_list.append(init_weight(rank, 
                                                rank, 
                                                weights = None, 
                                                init_scale = init_scale))

            # Left subspace factor 
            comp_layers_list.append(init_weight(rank, 
                                                config.hidden_size, 
                                                weights = U_init, 
                                                init_scale = 1.0))
            
            # Activation function: tanh
            comp_layers_list.append(nn.GELU())

        # Last layer classifier
        elif l == num_layers - 1:
            comp_layers_list.append(init_weight(config.hidden_size, 
                                                config.num_labels, 
                                                weights = None, 
                                                init_scale = init_scale))

        # Hidden layers
        else:
            # Low-rank weight matrix
            comp_layers_list.append(init_weight(rank, 
                                                rank, 
                                                weights = None, 
                                                init_scale = init_scale))

            # Activation function 
            comp_layers_list.append(nn.GELU())

    return comp_layers_list


# Helper function to re-parameterize RoBERTa classifier to be compress 
def reparameterize_roberta_classifier(model: CustomRobertaForSequenceClassification, 
                                      config: RobertaConfig,
                                      U_init: np.ndarray = None, 
                                      V_init: np.ndarray = None, 
                                      compress_rank: int = None, 
                                      init_scale: float = 1.0) -> List[nn.Module]:
            

    # Get compression rank - default is twice number of classes
    rank = compress_rank if compress_rank is not None else 2 * config.num_labels

    # Compress classifier depending on number of layers
    num_layers = model.classifier.num_layers

    # Two layers
    if num_layers == 2:
        comp_layers_list = reparameterize_two_layer_classifier(config,
                                                               U_init = U_init,
                                                               V_init = V_init, 
                                                               rank = rank, 
                                                               init_scale = init_scale)
                        
    # More than two layers
    else: 
        comp_layers_list = reparameterize_deep_classifier(config, 
                                                          U_init = U_init, 
                                                          V_init = V_init, 
                                                          rank = rank,
                                                          init_scale = init_scale)
        

    # Return compressed parameterization
    return comp_layers_list


'''
Compress Roberta classifier. Modifies `model' input in-place
'''
def compress_roberta_classifier(model: CustomRobertaForSequenceClassification,
                                config: RobertaConfig,
                                weights_init: List[np.ndarray] = None, 
                                grads_init: List[np.ndarray] = None,
                                compress_init_type: str = 'gradient',
                                compress_rank: int = None,
                                init_scale: float = 1.0) -> CustomRobertaClassificationHead:
    
    assert compress_init_type in ['gradient', 'random'], 'Please choose an inititliazation type in [gradient, random]'
    
    rank = compress_rank if compress_rank is not None else 2 * config.num_labels

    # Get gradients and weights after a single forwards and backwards pass (no updates)
    if compress_init_type == 'random': # Random subspace initialization
        U_init = None 
        V_init = None 
    else:
        assert weights_init is not None and grads_init is not None
        # Get left and right "good" subspace factors 
        G1_0 = grads_init[0] # Gradient of first-layer weights at initialization
        U_init, V_init = left_right_inits(weights_init, G1_0, rank, init_scale) 


    # Compressed classifier 
    comp_layers_list = reparameterize_roberta_classifier(model,
                                                         config,
                                                         U_init=U_init,
                                                         V_init=V_init,
                                                         compress_rank = rank)
    
    model.classifier.layers = nn.ModuleList(comp_layers_list)

    return model


'''
Get fully parameterized Roberta model 
'''
def get_model(num_classifier_layers: int = 2,
              init_scale: float = 1.0, 
              freeze_pretrained: bool = True) -> CustomRobertaForSequenceClassification:
    
    # Get original model 
    model = CustomRobertaForSequenceClassification.from_pretrained('roberta-base', 
                                                                   num_classifier_layers=num_classifier_layers,
                                                                   classifier_init_scale=init_scale)
    
    # Freeze pre-trained base layers, 
    #   so only train classifier head     
    if freeze_pretrained: 
        for name, param in model.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False

    return model 
    

