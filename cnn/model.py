import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.vgg import VGG, vgg16, vgg16_bn, VGG16_Weights, VGG16_BN_Weights


from copy import deepcopy
from train_utils import train
from utils import left_right_inits, left_right_inits_2

from typing import Optional, List

# Change classification head in VGG 
def _modify_vgg_classifier(model: VGG,
                           new_num_classes: int, 
                           activation: str, 
                           init_type: str, 
                           init_scale: float = 1.0,
                           pool_size: tuple = (3, 3)) -> VGG:
    
    model.classifier[0] = nn.Linear(512 * pool_size[0] * pool_size[1], 4096) # Change first linear layer to reflect new avgpool output size
    model.classifier[-1] = nn.Linear(4096, new_num_classes) # Change last linear layer so it has correct number of classes
    
    for (i, m) in enumerate(model.classifier): 
        # Replace all ReLU parts with another activation
        if isinstance(m, nn.ReLU):
            model.classifier[i] = getattr(nn, activation)()
        
        # Re-initialize linear weights so that they are scaled orthogonal matrices
        elif isinstance(m, nn.Linear): 
            if init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=init_scale)
            elif init_type == 'gaussian':
                nn.init.normal_(m.weight, mean=0.0, std=init_scale)
            elif init_type == 'uniform':
                nn.init.uniform_(m.weight, a = -1 * init_scale, b = 1 * init_scale)
            else:
                raise ValueError("Invalid initialization type.")
    
    return model


def get_custom_vgg16(cls_activation: str,
                     new_num_classes: int,
                     cls_init_type: str = 'orthogonal', 
                     cls_init_scale: float = 1.0,
                     weights: Optional[VGG16_Weights] = None, 
                     progress: bool = True,
                     batch_norm: bool = True, 
                     pool_size: tuple = (3, 3),
                     freeze_conv: bool = False, 
                     **kwargs) -> VGG:
        
    # Load in base model with original number of classes (1000)
    model = vgg16(weights, progress, num_classes=1000, **kwargs) if not batch_norm else vgg16_bn(weights, progress, num_classes=1000, **kwargs) 
    model.avgpool = nn.AdaptiveAvgPool2d(pool_size) # Change avgpool size to be smaller
    model = _modify_vgg_classifier(model, new_num_classes, cls_activation, cls_init_type, cls_init_scale, pool_size=pool_size) # Change classifier

    # Freeze convolutional layers if necessary
    if weights is not None and freeze_conv:
        for param in model.features.parameters():
            param.requires_grad=False
    
    return model


# Compressing VGG classifier 
def compress_vgg16_classifier(orig_model: VGG, 
                              rank: int, 
                              init_method: str,
                              init_scale: float,  
                              W_0: List[np.ndarray] = None,
                              U1_2r: np.ndarray = None, 
                              V1_2r: np.ndarray = None,
                              pool_size: tuple = (3, 3)) -> VGG:
    
     # Create a deepcopy of original model
    comp_model = deepcopy(orig_model)

    # Compress linear layers in classifier
    comp_model.classifier = nn.Sequential(
        nn.Linear(512 * pool_size[0] * pool_size[1], 2*rank),
        nn.Linear(2*rank, 2*rank),
        orig_model.classifier[1],
        orig_model.classifier[2],
        nn.Linear(2*rank, 2*rank),
        nn.Linear(2*rank, 4096),
        orig_model.classifier[4],
        orig_model.classifier[5],
        orig_model.classifier[6]
    )

    # Initialize compressed classifier weights
    with torch.no_grad(): 
        # Initialize as update subspaces
        if U1_2r is not None and V1_2r is not None: 
            # Initialize V factor
            comp_model.classifier[0].weight.copy_(torch.Tensor(V1_2r).transpose(0, 1))

            # Calculate U factor
            U2_2r = W_0[1] @ U1_2r / init_scale
            
            # Initialize U factor
            comp_model.classifier[5].weight.copy_(torch.Tensor(U2_2r))

        # Initialize as random subspaces
        else: 
            nn.init.orthogonal_(comp_model.classifier[0].weight, gain=1)
            nn.init.orthogonal_(comp_model.classifier[5].weight, gain=1)

    # Initialize other compressed weights
    if init_method == 'orthogonal':
        nn.init.orthogonal_(comp_model.classifier[1].weight, gain=init_scale)
        nn.init.orthogonal_(comp_model.classifier[4].weight, gain=init_scale)
    elif init_method == 'gaussian':
        nn.init.normal_(comp_model.classifier[1].weight, mean=0.0, std=init_scale)
        nn.init.normal_(comp_model.classifier[4].weight, mean=0.0, std=init_scale)
    elif init_method == 'uniform':
        nn.init.uniform_(comp_model.classifier[1].weight, a = -init_scale, b = init_scale)
        nn.init.uniform_(comp_model.classifier[4].weight, a = -init_scale, b = init_scale)
    else:
        raise ValueError("Invalid initialization type.")

    return comp_model
    



def compress_vgg16(orig_model: VGG, 
                   rank: int, 
                   init_method: str,
                   init_scale: float, 
                   criterion: nn.Module,
                   train_set: Dataset, 
                   batch_size: int, 
                   subspace_type: int, 
                   device: str, 
                   pool_size: tuple = (3, 3)) -> VGG:
    
    # Initialize U and V factors as random subspaces
    if subspace_type is None:
        return compress_vgg16_classifier(orig_model, rank, init_method, init_scale, pool_size=pool_size)
    
    # Calculate subspace where update occurs
    if batch_size is None:
        batch_size = train_set.data.shape[0]
    dummy_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    dummy_optimizer = torch.optim.SGD(orig_model.parameters())

    # Subspace type
    if subspace_type == 1:
        results = train(orig_model, 
                        criterion, 
                        dummy_optimizer, 
                        dummy_loader, 
                        dummy_loader, 
                        1, 
                        device, 
                        num_inner_iter=1,
                        return_weights=True, 
                        return_grads=True)
        
        # Get first-layer weight and gradient at initialization
        W_0 = results['all_weights'][0]
        G_0 = results['all_grads'][0]

        W1_0 = W_0[0]
        G1_0 = G_0[0]
        
        # Get subspace where "small" updates occur
        U1, V1 = left_right_inits(W_0, G1_0, rank, init_scale)
    
    elif subspace_type == 2:
        results = train(orig_model, 
                        criterion, 
                        dummy_optimizer, 
                        dummy_loader, 
                        dummy_loader, 
                        1, 
                        device, 
                        num_inner_iter=1,
                        return_weights=True, 
                        return_grads=False, 
                        return_outs=True)
        
        # Get input to classifier
        X_est, Y = next(iter(dummy_loader)) # Data matrix
        H_est = orig_model.avgpool(orig_model.features(X_est.to(device))) # Inputs to classifier
        H_est = torch.flatten(H_est, 1).detach().cpu().numpy().T

        # Get labels
        ZL_0 = results['all_outs'][0].T # Model output of first epoch
        E_0 = ZL_0 - Y.detach().cpu().numpy().T # Residual

        # Get subspace where "small" updates occur
        W_0 = results['all_weights'][0]
        U1, V1 = left_right_inits_2(W_0, E_0, H_est, rank, init_scale)
    
    else:
        raise ValueError("Invalid subspace type.")



    return compress_vgg16_classifier(orig_model, rank, init_method, init_scale, W_0=W_0, U1_2r=U1, V1_2r=V1, pool_size=pool_size)

