import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import tqdm
import argparse

from train_utils import train
from model import get_model, compress_roberta_classifier
from dataset_utils import get_glue_dataloaders
from metrics import GlueMetric
from utils import * 


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # Data 
    parser.add_argument('--glue_task_name', type=str, choices=['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2'])
    parser.add_argument('--data_dir', type=str, default='/scratch/qingqu_root/qingqu1/alecx/')
    
    # Model
    parser.add_argument('--num_classifier_layers', type=int, default=2)
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--subspace_type', type=str, default='gradient', choices=['gradient', 'random'], help='Type of ``update subspace`` for compressed classifier.')
    parser.add_argument('--compress_rank', type=int, default=None)

    # Initialization method
    parser.add_argument('--init_scale', type=float, default=1e-2, help='Initialization scale')
    
    # Training settings
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Max training epochs')
    parser.add_argument('--accum_steps', type=int, default=None, help='Number of gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')

    # Save directory
    parser.add_argument('--parent_save_dir', type=str, default='./results/')

    args = parser.parse_args()

    return args


# Main function
def main():
    args = parse_args() # Get command line args
    print("args = ", args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Device = ', device)

    # Input data
    glue_task_name = args.glue_task_name
    #data_dir = os.path.join(args.data_dir, glue_task_name)

    # Network
    num_classifier_layers = args.num_classifier_layers
    compress_ = args.compress
    compress_rank = args.compress_rank
    subspace_type = args.subspace_type
    init_scale = args.init_scale

    # Training
    num_trials = args.num_trials
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr


    # Define save dir
    parent_save_dir = args.parent_save_dir
    data_exp_dir = os.path.join(parent_save_dir, glue_task_name)
    model_exp_dir = 'roberta_init_scale_' + str(init_scale) + '_eta_' + str(lr)
    if compress_:
        model_type_dir = 'compressed_classifier_rank_' + str(compress_rank) + '_subspace_' + str(subspace_type) + '_' + str(num_classifier_layers) + '_layers' 
    else: 
        model_type_dir = 'orig_classifier_' + str(num_classifier_layers) + '_layers'
    
    save_dir = os.path.join(parent_save_dir, data_exp_dir, model_exp_dir, model_type_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # Run training trials
    for i in tqdm.tqdm(range(num_trials)):
        print("===== TRIAL {} =====".format(i + 1))

        # Generate train samples
        train_loader, val_loader = get_glue_dataloaders(glue_task_name, batch_size=batch_size)
        accum_steps = args.accum_steps if args.accum_steps is not None else len(train_loader) # Default: number of accumulation steps = number of batches 

        # Create object to compute metrics
        metric = GlueMetric(glue_task_name)

        # Get fully parameterized model 
        model = get_model(num_classifier_layers=num_classifier_layers, init_scale=init_scale)
        model = model.to(device)

        # Compress the model if needed
        if compress_: 
            if subspace_type == 'gradient': # Gradient-based initialization
                dummy_optimizer = AdamW(model.parameters(), lr=lr)
                # Perform one training step 
                results = train(model, 
                                train_loader, 
                                val_loader, 
                                metric, 
                                1, # One epoch 
                                len(train_loader), # Number of gradient accumulation steps, make it length of loader to get most accurate gradient
                                device, 
                                dummy_optimizer, 
                                return_weights=True, 
                                return_grads=True)
            
                # Get initialized weights and gradients
                weights_init = results['weights'][0]
                grads_init = results['grads'][0]

            else: # Random subspace initialization
                weights_init, grads_init = None, None

            model = compress_roberta_classifier(model, 
                                                model.config, 
                                                weights_init, 
                                                grads_init,
                                                compress_init_type=subspace_type, 
                                                compress_rank=compress_rank,
                                                init_scale=init_scale)
            
            model = model.to(device)


        # Print new classifier and number of parameters
        print("New Roberta classifier = ", model.classifier)

        num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print("Number of trainable parameters = ", num_params)

        # Set up optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.0) #1e-4)

        total_steps = epochs * (len(train_loader) // accum_steps) # Calculate total number of optimization steps
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, 
        #     0.05 * total_steps, 
        #     total_steps
        # ) # Linear schedular
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            0.03 * total_steps, 
            total_steps, 

        )

        # Train classification head
        results = train(
            model, 
            train_loader, 
            val_loader, 
            metric, 
            epochs, 
            accum_steps,
            device, 
            optimizer, 
            scheduler
        )

        # Get and store results
        trial_train_losses = results['train_losses']
        trial_train_metrics = results['train_metrics']

        trial_val_losses = results['val_losses']
        trial_val_metrics = results['val_metrics']        


        # Save trial results
        np.savez(os.path.join(save_dir, 'trial_' + str(i + 1) + '_losses'), train_losses=trial_train_losses, val_losses=trial_val_losses)
        np.savez(os.path.join(save_dir, 'trial_' + str(i + 1) + '_accs'), train_accs=trial_train_metrics, val_accs=trial_val_metrics)



if __name__ == '__main__':
    main()

    


        