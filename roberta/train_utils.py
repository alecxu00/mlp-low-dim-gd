import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import Dict
import tqdm
import logging 

from model import CustomRobertaForSequenceClassification
from metrics import GlueMetric 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

'''
Helper function to store classifier gradients 
'''
def get_epoch_grads(model: CustomRobertaForSequenceClassification):
    grads = []

    for param in model.classifier.parameters():
        grads.append(param.grad.detach().cpu().numpy())
        
    return grads 


'''
Helper function to store classifier weights 
'''
def get_epoch_weights(model: CustomRobertaForSequenceClassification):
    return model.classifier.get_weights()


'''
Perform one training epoch 
'''
def train_epoch(model: CustomRobertaForSequenceClassification,
                accum_steps: int, 
                train_loader: DataLoader, 
                metric: GlueMetric, 
                device: str, 
                optimizer: nn.Module, 
                scheduler: nn.Module, 
                return_weights: bool = False, 
                return_grads: bool = False) -> Dict:
    
    model.train()

    epoch_results = {}
    epoch_loss = 0.0

    metric.clear()
    for i, batch in enumerate(train_loader):
        # Get inputs to model 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Get model outputs
        outputs = model(input_ids=input_ids, 
                      attention_mask=attention_mask, 
                      labels=labels)
        
        # Get metric
        preds = torch.argmax(torch.softmax(outputs.logits, dim=-1), dim=-1)
        metric.add_batch(preds, labels)
        
        # Get loss and backprop
        loss = outputs.loss
        loss /= accum_steps
        epoch_loss += loss.item() # Update running loss 
        loss.backward()

        if (i + 1) % accum_steps == 0:
            # Save gradients 
            if return_grads:
                epoch_results['grads'] = get_epoch_grads(model)

            # Take optimization and scheduler step 
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Zero-out gradients
            optimizer.zero_grad()

            # Save updated weights
            if return_weights:
                epoch_results['weights'] = get_epoch_weights(model)

    epoch_results['train_loss'] = epoch_loss 
    epoch_results['train_metrics'] = metric.compute()

    return epoch_results

    
def eval_(model: CustomRobertaForSequenceClassification,
          val_loader: DataLoader, 
          metric: GlueMetric, 
          device: str) -> Dict:
    
    model.eval()

    val_results = {}
    val_loss = 0.0 

    metric.clear()
    for batch in val_loader:
         # Get inputs to model 
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Get model outputs
        outputs = model(input_ids=input_ids, 
                      attention_mask=attention_mask, 
                      labels=labels)
        
        # Get loss and backprop
        loss = outputs.loss
        val_loss += loss.item()

        # Get metric
        preds = torch.argmax(torch.softmax(outputs.logits, dim=-1), dim=-1)
        metric.add_batch(preds, labels)
    
    val_results['val_loss'] = val_loss
    val_results['val_metrics'] = metric.compute()

    return val_results


def train(model: CustomRobertaForSequenceClassification,
          train_loader: DataLoader, 
          val_loader: DataLoader, 
          metric: GlueMetric, 
          epochs: int, 
          accum_steps: int, 
          device: str,
          optimizer: nn.Module, 
          scheduler: nn.Module = None,
          return_weights: bool = False, 
          return_grads: bool = False) -> Dict:
    
    results = {'train_losses': [],
               'train_metrics': [],
               'val_losses': [],
               'val_metrics': [],
               'weights': [model.classifier.get_weights()], 
               'grads': []}
    
    for epoch in tqdm.tqdm(range(epochs)):
        # Perform training epoch 
        epoch_results = train_epoch(model, 
                                    accum_steps,
                                    train_loader, 
                                    metric,
                                    device, 
                                    optimizer, 
                                    scheduler, 
                                    return_weights=return_weights,
                                    return_grads=return_grads)
        
        # Print and store train epoch results
        logging.info("Finish Epoch {}".format(epoch))
        logging.info("Train loss: {}, train metrics: {}".format(epoch_results['train_loss'], epoch_results['train_metrics']))

        results['train_losses'].append(epoch_results['train_loss'])
        results['train_metrics'].append(epoch_results['train_metrics']) 
        if return_weights:
            results['weights'].append(epoch_results['weights'])
        if return_grads:
            results['grads'].append(epoch_results['grads'])

        # Validation
        val_results = eval_(model, 
                            val_loader, 
                            metric, 
                            device)

        # Print and store val results
        logging.info("Val loss: {}, val metrics: {}".format(val_results['val_loss'], val_results['val_metrics']))
        results['val_losses'].append(val_results['val_loss'])
        results['val_metrics'].append(val_results['val_metrics'])


    return results

