import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import tqdm
import logging
import psutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Training
def train_epoch(model, 
                train_loader, 
                optimizer, 
                criterion, 
                device, 
                scheduler=None,
                update=True,
                return_weights=False,
                return_grads=False,
                return_outs=False):
    
    model.train()

    running_loss = 0.0
    num_correct = 0
    
    if return_weights:
        weights = []
    if return_grads:
        grads = []
    
    N = 0
    for data, targets in train_loader:
        # Get model predictions
        data = data.type(torch.FloatTensor)
        targets = targets.type(torch.FloatTensor)
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        out, _ = model(data)
        
        # Backwards pass
        loss = criterion(out, targets)
        loss.backward()
        
        # Get model gradients
        if return_grads:
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach().cpu().numpy()) 
        
        # Update and store model weights
        if update:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if return_weights:
            for layer in model.layers:
                if isinstance(layer, nn.Linear):
                    weights.append(layer.weight.detach().cpu().numpy())
                elif isinstance(layer, nn.Parameter):
                    weights.append(layer.data.cpu().numpy())
                    

        # Compute train performance
        running_loss += loss.item() # Loss
        
        pred = torch.argmax(out, dim=1) # Accuracy 
        if targets.shape == pred.shape:
            num_correct += torch.sum(pred==targets).item()
        else: # targets are one-hot encoded
            num_correct += torch.sum(pred==torch.argmax(targets, dim=1)).item()
            
        
        N += data.shape[0]
        
        del loss
        if not return_outs: del out


    epoch_loss = running_loss #/ len(train_loader.sampler)
    epoch_accuracy = 100. * num_correct / N

    epoch_results = {'epoch_loss': epoch_loss,
                     'epoch_accuracy': epoch_accuracy}
    
    if return_weights:
        epoch_results['epoch_weights'] = weights
    if return_grads:
        epoch_results['epoch_grads'] = grads
    if return_outs:
        epoch_results['epoch_outs'] = out.detach().cpu().numpy()

    return epoch_results


def eval_(model, val_loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    num_correct = 0
    N = 0
    with torch.no_grad():
        for data, targets in val_loader:
            # Forward pass
            data = data.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)
            data, targets = data.to(device), targets.to(device)
            out, _ = model(data)
            
            # Get val loss
            loss = criterion(out, targets)
            
            # Get metrics
            running_loss += loss.item() # Loss
            
            N += data.shape[0] # Accuracy
            pred = torch.argmax(out, dim=1)
            if targets.shape == pred.shape:
                num_correct += torch.sum(pred==targets).item()
            else: # targets are one-hot encoded
                num_correct += torch.sum(pred==torch.argmax(targets, dim=1)).item()
            
            
    val_loss = running_loss #/ len(val_loader.sampler)
    val_accuracy = 100. * num_correct / N
    
    val_results = {'val_loss': val_loss, 'val_accuracy': val_accuracy}
    return val_results

def train(model, 
          criterion,
          optimizer, 
          train_loader, 
          val_loader, 
          epochs, 
          device, 
          scheduler=None, 
          return_weights=False,
          return_grads=False, 
          return_outs=False):
    
    model = model.to(device)
    
    # Get initial model weights
    weights_init = []
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            weights_init.append(layer.weight.detach().cpu().numpy())
        elif isinstance(layer, nn.Parameter):
            weights_init.append(layer.data.cpu().numpy())
    
    
    if return_weights:
        all_weights = [weights_init]
    if return_grads:
        all_grads = []
    if return_outs:
        all_outs = []

    # Training
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in tqdm.tqdm(range(1, epochs+1)):
        epoch_results = train_epoch(model, 
                                    train_loader,
                                    optimizer, 
                                    criterion, 
                                    device, 
                                    scheduler=scheduler,
                                    return_weights=return_weights,
                                    return_grads=return_grads,
                                    return_outs=return_outs)

        if return_weights:
            all_weights.append(epoch_results['epoch_weights'])
        if return_grads:
            all_grads.append(epoch_results['epoch_grads'])
        if return_outs:
            all_outs.append(epoch_results['epoch_outs'])

        train_losses.append(epoch_results['epoch_loss'])
        train_accs.append(epoch_results['epoch_accuracy'])

        # Print training progress
        #logging.info("Finish Epoch {}".format(epoch))
        #logging.info("Train loss: {}, train accuracy: {}".format(epoch_results['epoch_loss'], epoch_results['epoch_accuracy']))
        
        #val_results = eval_(model, val_loader, criterion, device) 
        #val_losses.append(val_results['val_loss'])
        #val_accs.append(val_results['val_accuracy'])
        #logging.info("Val loss: {}, val accuracy: {}".format(val_results['val_loss'], val_results['val_accuracy']))
                    
    results = {'train_losses': train_losses,
               'train_accs': train_accs,
               'val_losses': val_losses,
               'val_accs': val_accs}
    
    if return_weights:
        results['all_weights'] = all_weights
    if return_grads:
        results['all_grads'] = all_grads
    if return_outs:
        results['all_outs'] = all_outs

    return results