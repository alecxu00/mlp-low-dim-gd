import os
import numpy as np
import torch
import torch.nn as nn

from utils import * 
from model import get_custom_vgg16, compress_vgg16, compress_vgg16_classifier
from train_utils import train
from data import get_cifar10, get_cifar100

import tqdm
import argparse


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # Data 
    parser.add_argument('--data_type', type=str, default='cifar10', choices=['cifar10', 'cifar100'])

    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--train_samples_per_class', type=int, default=5000)
    parser.add_argument('--val_samples_per_class', type=int, default=1000)
    parser.add_argument('--one_hot', action='store_true')
    parser.add_argument('--data_dir', type=str, default='/scratch/qingqu_root/qingqu1/alecx/')
    
    # Model
    parser.add_argument('--activation', type=str, default='ELU', help='Activation function',
                        choices=['ELU', 'GELU', 'SiLU', 'ReLU'])
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--subspace_type', type=int, default=None, choices=[1, 2, None], help='Type of ``update subspace`` for compressed model. "None" means random subspace.')
    parser.add_argument('--compress_rank', type=int, default=None)
    parser.add_argument('--pt_conv', action='store_true')
    parser.add_argument('--freeze_conv', action='store_true')

    # Initialization method
    parser.add_argument('--init_method', type=str, default='orthogonal', choices=['orthogonal', 'gaussian', 'uniform'])
    parser.add_argument('--init_scale', type=float, default=1e-2, help='Initialization scale')
    
    # Training settings
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--epochs', type=int, default=500, help='Max training epochs')
    parser.add_argument('--loss', type=str, default='MSELoss', choices=['MSELoss', 'CrossEntropyLoss'], help='Training loss')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'])

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
    data_type = args.data_type
    K = args.num_classes
    N_k = args.train_samples_per_class
    N = N_k * K
    N_k_val = args.val_samples_per_class
    one_hot = args.one_hot
    data_dir = os.path.join(args.data_dir, data_type)

    # Network
    act_str = args.activation
    pt_conv = args.pt_conv
    freeze_conv = args.freeze_conv
    compress_ = args.compress
    compress_rank = args.compress_rank if args.compress_rank is not None else K 
    subspace_type = args.subspace_type

    init_method = args.init_method
    init_scale = args.init_scale

    # Training
    num_trials = args.num_trials
    batch_size = args.batch_size
    shuffle = args.shuffle
    epochs = args.epochs
    loss_fn = args.loss
    lr = args.lr
    opt_type = args.optimizer


    # Define save dir
    parent_save_dir = args.parent_save_dir
    data_exp_dir = data_type + '_K_' + str(K) + '_N_' + str(N)

    if freeze_conv:
        model_exp_dir = act_str + '_vgg16_frozen_conv_' + init_method + '_init_scale_' + str(init_scale) + '_eta_' + str(lr)
    else:
        model_exp_dir = act_str + '_vgg16_' + init_method + '_init_scale_' + str(init_scale) + '_eta_' + str(lr)
    model_type_dir = 'compressed_model_rank_' + str(compress_rank) + '_subspace_' + str(subspace_type) if compress_ else 'orig_model'
    
    
    save_dir = os.path.join(parent_save_dir, data_exp_dir, model_exp_dir, model_type_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    

    # Run training trials
    all_train_losses = np.zeros((num_trials, epochs))
    all_train_accs = np.zeros((num_trials, epochs))

    all_val_losses = np.zeros((num_trials, epochs))
    all_val_accs = np.zeros((num_trials, epochs))


    for i in tqdm.tqdm(range(num_trials)):
        print("===== TRIAL {} =====".format(i + 1))

        # Generate train samples
        if data_type == 'cifar10':
            train_set, train_loader = get_cifar10(K, N_k, data_dir, batch_size=batch_size, train=True, one_hot=one_hot, shuffle=shuffle)
            _, val_loader = get_cifar10(K, N_k_val, data_dir, batch_size=batch_size, train=False, one_hot=one_hot, shuffle=shuffle)
        elif data_type == 'cifar100':
            train_set, train_loader = get_cifar100(K, N_k, data_dir, batch_size=batch_size, train=True, one_hot=one_hot, shuffle=shuffle)
            _, val_loader = get_cifar100(K, N_k_val, data_dir, batch_size=batch_size, train=False, one_hot=one_hot, shuffle=shuffle)
        else:
            raise ValueError("Invalid dataset.")
        
        # Loss function
        criterion = getattr(nn, loss_fn)(reduction='mean')

        # Get fully parameterized model 
        if pt_conv: # Load pre-trained conv layer weights
            model = get_custom_vgg16(act_str, K, cls_init_type=init_method, cls_init_scale=init_scale, weights="IMAGENET1K_V1", freeze_conv=freeze_conv)
        else:
            model = get_custom_vgg16(act_str, K, cls_init_type=init_method, cls_init_scale=init_scale, weights=None, freeze_conv=freeze_conv)

        # Compress the model if needed
        if compress_: 
            model = compress_vgg16(model, compress_rank, init_method, init_scale, criterion, train_set, batch_size, subspace_type, device)
        
        # Print new classifier and number of parameters
        print("New VGG classifier = ", model.classifier)

        num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print("Number of trainable parameters = ", num_params)

        # Train model
        optimizer = getattr(torch.optim, opt_type)(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
        results = train(model, criterion, optimizer, train_loader, val_loader, epochs, device, scheduler=scheduler)

        # Get and store results
        trial_train_losses = results['train_losses']
        trial_train_accs = results['train_accs']

        trial_val_losses = results['val_losses']
        trial_val_accs = results['val_accs']        


        all_train_losses[i, :] = trial_train_losses
        all_train_accs[i, :] = trial_train_accs

        all_val_losses[i, :] = trial_val_losses
        all_val_accs[i, :] = trial_val_accs


        # Save trial results
        np.savez(os.path.join(save_dir, 'trial_' + str(i + 1) + '_losses'), train_losses=trial_train_losses, val_losses=trial_val_losses)
        np.savez(os.path.join(save_dir, 'trial_' + str(i + 1) + '_accs'), train_accs=trial_train_accs, val_accs=trial_val_accs)


    # Save all results
    np.savez(os.path.join(save_dir, 'losses_all_trials'), train_losses=all_train_losses, val_losses=all_val_losses)
    np.savez(os.path.join(save_dir, 'accs_all_trials'), train_accs=all_train_accs, val_accs=all_val_accs)



if __name__ == '__main__':
    main()

    


        