import numpy as np
import torch
import torch.nn as nn
from typing import List
from train_utils import train
from torch.utils.data import DataLoader, Dataset

from utils import left_right_inits, svd_

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x: torch.Tensor):
        return torch.abs(x)

class SmoothAbs(nn.Module):
    def __init__(self, k: float = 10):
        super(SmoothAbs, self).__init__()

        self.k = torch.Tensor([k])
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        self.k = self.k.to(x.device)
        return (2 / self.k) * self.softplus(x) - x - (2 / self.k) * torch.log(torch.Tensor([2]).to(x.device)) 


class CustomSoftplus(nn.Module):
    def __init__(self, beta: float = 1.0):
        super(CustomSoftplus, self).__init__()
        self.beta = torch.Tensor([beta])
        self.softplus = nn.Softplus()
    
    def forward(self, x: torch.Tensor):
        self.beta = self.beta.to(x.device)
        
        return (1 / self.beta) * self.softplus(x) - torch.log(torch.Tensor([2]).to(x.device)) 


class CustomSiLU(nn.Module):
    def __init__(self):
        super(CustomSiLU, self).__init__()
        self.silu = nn.SiLU()
        
    def forward(self, x: torch.Tensor):
        return (self.silu(x) * (x <= 0).float()) + ((x/2) * (x > 0).float())
    
class LeakySoftplus(nn.Module):
    def __init__(self, slope=0.01):
        super(LeakySoftplus, self).__init__()
        self.softplus = nn.Softplus()
        self.slope = slope
    
    def forward(self, x: torch.Tensor): 
        out1 = self.slope * x
        out2 = (1 - self.slope) * ( self.softplus(x) - torch.log(torch.Tensor([2]).to(x.device)) )
        return out1 + out2
    
activations_map = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'LeakySoftplus': LeakySoftplus(),
    'RReLU': nn.RReLU(),
    'Abs': Abs(),
    'GELU': nn.GELU(),
    'ELU': nn.ELU(alpha=1),
    'SiLU': nn.SiLU(),
    'Mish': nn.Mish(),
    'Softplus': nn.Softplus(),
    'CustomSoftplus': CustomSoftplus(),
    'CustomSiLU': CustomSiLU(),
    'CELU': nn.CELU(),
    'SmoothAbs': SmoothAbs(),
    'tanh': nn.Tanh(),
    'Identity': None
}    

class MLP(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 in_dim: int, 
                 hidden_dim: int, 
                 num_classes: int,
                 activation: str,
                 freeze_last_layer=False,
                 diff_init_last_layer=False,
                 bias=False, 
                 batchnorm=False, 
                 residual=False, 
                 init_method='orthogonal',
                 last_layer_init_method='orthogonal',
                 init_scale: float = 1.0,
                 last_layer_init_scale: float = 1.0):

        super(MLP, self).__init__()
        
        # Dimensions of weight matrices
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = num_classes
        self.bias = bias
        self.freeze_last_layer = freeze_last_layer
        self.diff_init_last_layer = diff_init_last_layer
        
        # Number of layers
        assert num_layers > 1
        self.num_layers = num_layers

        # Create layers
        self.activation_type = activation
        self.activation = activations_map[activation]
        self.batchnorm = batchnorm
        self.residual = residual
        self.layers = self._create_layers()
        
        if self.freeze_last_layer:
            for param in self.layers[-1].parameters():
                param.requires_grad = False
        
        # Initialize weights
        self.init_method = init_method
        self.last_layer_init_method = last_layer_init_method
        self.init_scale = init_scale
        self.last_layer_init_scale = last_layer_init_scale
        self._init_weights()
    

    # Helper function to create layers
    def _create_layers(self):
        # First layer
        layers_list = []
        for l in range(self.num_layers):
            # First layer
            if l == 0: 
                layers_list.append(nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias) )

                if self.activation is not None:
                    layers_list.append(self.activation)

                if self.batchnorm:
                    layers_list.append( nn.BatchNorm1d(self.hidden_dim, affine=False) )
                    
            # Last layer (linear classifier)
            elif l == self.num_layers - 1:
                layers_list.append( nn.Linear(self.hidden_dim, self.out_dim, bias=self.bias) )
            
            # Hidden layers
            else:
                layers_list.append( nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias) )
                
                if self.activation is not None:
                    layers_list.append(self.activation)
                
                if self.batchnorm:
                    layers_list.append( nn.BatchNorm1d(self.hidden_dim, affine=False) )
        
        return nn.ModuleList(layers_list)

    
    # Helper function to initialize weights
    def _init_weights(self):
        print(f"Initializing weights using {self.init_method} method with scale {self.init_scale}")

        for (l, layer) in enumerate(self.layers):                    
            if isinstance(layer, nn.Linear):
                # Freeze last layer at initialization, so intialize it as something else
                if l == (len(self.layers) - 1) and ( self.freeze_last_layer or self.diff_init_last_layer ):
                    # Orthogonal
                    if self.last_layer_init_method == 'orthogonal':
                        nn.init.orthogonal_(layer.weight, gain=self.last_layer_init_scale) #self.init_scale)
                                    
                    # Gaussian
                    elif self.last_layer_init_method == 'gaussian':
                        nn.init.normal_( layer.weight, mean=0.0, std=self.last_layer_init_scale) #self.init_scale / np.sqrt(self.hidden_dim) )
                        
                    # Uniform
                    elif self.last_layer_init_method == 'uniform':
                        nn.init.uniform_(layer.weight, a = -self.last_layer_init_scale, b=self.last_layer_init_scale)
                        
                    # Identity
                    elif self.last_layer_init_method == 'identity':
                        nn.init.eye_(layer.weight)

                    else:
                        raise ValueError("Invalid last layer initialization type.")
                
                # Initialize all other layers as is
                else:
                    # Orthogonal
                    if self.init_method == 'orthogonal':
                        nn.init.orthogonal_(layer.weight, gain=self.init_scale)
                                    
                    # Gaussian
                    elif self.init_method == 'gaussian':
                        nn.init.normal_( layer.weight, mean=0.0, std=self.init_scale / np.sqrt(self.hidden_dim) )
                        
                    # Uniform
                    elif self.init_method == 'uniform':
                        nn.init.uniform_(layer.weight, a = -1 * self.init_scale, b = self.init_scale)
                        
                    # Identity
                    elif self.init_method == 'identity':
                        nn.init.eye_(layer.weight)

                    else:
                        raise ValueError("Invalid initialization type.")
    

    # Forward pass
    def forward(self, x):
        out = x
        for (l, layer) in enumerate(self.layers):
            out = out + layer(out) if (self.residual and l < len(self.layers) - 1) else layer(out) 
            
            if l == len(self.layers) - 2: # Output of 
                hidden_features = out
        
        return out, hidden_features
    

class CompressedMLP(nn.Module):
    def __init__(self, 
                 orig_init_weights: List[np.array], 
                 U: np.array, 
                 V: np.array, 
                 rank: int, 
                 num_classes: int, 
                 activation: str, 
                 init_method: str = 'orthogonal',
                 init_scale: float = 1.0, 
                 bias: bool = False):

        # Here, "U" and "V" are the U_{L-1} and V_1 factors

        super(CompressedMLP, self).__init__()
        
        self.U = torch.Tensor(U) # Of shape (d_h, 2r)
        self.V = torch.Tensor(V) # Of shape (d, 2r)
        self.W_L = torch.Tensor(orig_init_weights[-1]) # Of shape (K, d_h)

        self.hidden_dim = orig_init_weights[-1].shape[1] # Input dim of the last layer
        self.num_layers = len(orig_init_weights)
        self.rank = rank
        self.activation = activations_map[activation]
        self.num_classes = num_classes
        
        self.init_method = init_method
        self.init_scale = init_scale
        self.bias = bias
        
        self.layers, self.comp_params, self.subspace_params = self._create_layers()
        
    
    def _create_layers(self):        
        param_list = []
        
        comp_param_list = []
        subspace_param_list = []
        for l in range(self.num_layers - 1):
            layer = []
            
            if l == 0: 
                # Network V factor
                V_param = self._create_linear_layer(self.V)
                layer.append(V_param)
                subspace_param_list.append(V_param)
                
                # Compressed weights
                W_comp = nn.Linear(2*self.rank, 2*self.rank, bias=self.bias)
                self._init_weight(W_comp)
                
                layer.append(W_comp)
                comp_param_list.append(W_comp)
                
            elif l == self.num_layers - 2:
                # Compressed weights
                W_comp = nn.Linear(2*self.rank, 2*self.rank, bias=self.bias)
                self._init_weight(W_comp)
                
                layer.append(W_comp)
                comp_param_list.append(W_comp)
                
                # Network U factor
                U_param = self._create_linear_layer(self.U.transpose(0, 1))
                layer.append(U_param)
                subspace_param_list.append(U_param)
            
            else:
                # Compressed weights
                W_comp = nn.Linear(2*self.rank, 2*self.rank, bias=self.bias)
                self._init_weight(W_comp)
                
                layer.append(W_comp)
                comp_param_list.append(W_comp)
            
            
            # Activation function
            layer.append(self.activation)

            param_list.extend(layer)
            
        
        # Last layer: keep as is
        W_L = self._create_linear_layer(self.W_L.transpose(0, 1))

        param_list.append(W_L)
        comp_param_list.append(W_L)
        
        return nn.ParameterList(param_list), nn.ParameterList(comp_param_list), nn.ParameterList(subspace_param_list)
    
    
    def _init_weight(self, layer: nn.Linear):
        if self.init_method == 'orthogonal':
            nn.init.orthogonal_(layer.weight, gain=self.init_scale)
            
        elif self.init_method == 'uniform':
            nn.init.uniform_(layer.weight, a = -1 * self.init_scale, b = self.init_scale) 
            
        elif self.init_method == 'gaussian':
            nn.init.normal_(layer.weight, std=self.init_scale)

        else:
            raise ValueError("Invalid initialization type.")
            
    
    def _create_linear_layer(self, W: torch.Tensor):
        l = nn.Linear(W.shape[0], W.shape[1], bias=self.bias)
        with torch.no_grad():
            l.weight.copy_(W.transpose(0, 1))
            
        return l
        
    
    def forward(self, x: torch.Tensor):  
        out = x
        for (l, layer) in enumerate(self.layers):
            out = layer(out)
            
            if l == len(self.layers) - 2:
                hidden_features = out
        
        return out, hidden_features


# Function to compress a fully parameterized model
def compress_model(orig_model: MLP, 
                   rank: int, 
                   num_classes: int, 
                   init_method: str,
                   init_scale: float,
                   criterion: nn.Module, 
                   train_set: Dataset, 
                   subspace_type: str,
                   device: str,
                   angle: float = 0.0):
    
    # Single gradient step
    dummy_loader = DataLoader(train_set, batch_size=train_set.data.shape[0], shuffle=False)
    dummy_optimizer = torch.optim.SGD(orig_model.parameters())
        
    # Subspace type
    if subspace_type == 'gradient':
        results = train(orig_model, 
                        criterion, 
                        dummy_optimizer, 
                        dummy_loader, 
                        dummy_loader, 
                        1, 
                        device, 
                        return_weights=True, 
                        return_grads=True, 
                        return_outs=False)
        
        W_0 = results['all_weights'][0]
        G1_0 = results['all_grads'][0][0]
        
        U, V = left_right_inits(W_0, G1_0, rank, init_scale, angle=angle)

    elif subspace_type == 'random':
        results = train(orig_model, 
                        criterion, 
                        dummy_optimizer, 
                        dummy_loader, 
                        dummy_loader, 
                        1, 
                        device, 
                        return_weights=True, 
                        return_grads=False, 
                        return_outs=False)
        
        W_0 = results['all_weights'][0]
        
        V = svd_(np.random.randn(2*rank, W_0[0].shape[1]), full_matrices=False)[-1]
        U = svd_(np.random.randn(W_0[-2].shape[0], 2*rank), full_matrices=False)[0]

    else:
        raise ValueError("Invalid subspace type.")

    model = CompressedMLP(results['all_weights'][0], U, V, rank, num_classes, orig_model.activation_type, init_method=init_method, init_scale=init_scale)

    return model
