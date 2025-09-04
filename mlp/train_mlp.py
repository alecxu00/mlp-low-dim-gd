import os
import numpy as np
import torch
import torch.nn as nn

from utils import * 
from model import MLP, CompressedMLP, CompressedMLP2, compress_model
from train_utils import train
from data import generate_gmm_data, get_fashion_mnist

import tqdm
import argparse


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # Data 
    parser.add_argument('--data_type', type=str, default='synthetic', choices=['synthetic', 'fashion_mnist'])
    parser.add_argument('--data_dim', type=int, default=32, help='Dimension of data.')
    parser.add_argument('--data_rank', type=int, default=None, help='Rank of data matrix. Only used when data_type == synthetic')
    parser.add_argument('--whiten', action='store_true', help='Whiten data matrix. Only used when data_type == synthetic.')
    parser.add_argument('--normalize', action='store_true', help='Make data samples unit norm. Only used when data_type == synthetic and whiten_ is false.')


    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--train_samples_per_class', type=int, default=5000)
    parser.add_argument('--val_samples_per_class', type=int, default=1000)
    parser.add_argument('--data_dir', type=str, default='/scratch/qingqu_root/qingqu1/alecx/')
    

    # Model
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--activation', type=str, default='ELU', help='Activation function',
                        choices=['ELU', 'GELU', 'SiLU', 'ReLU'])
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--compress_rank', type=int, default=None)
    parser.add_argument('--subspace_type', type=int, default=None, choices=[1, 2, None], help='Type of ``update subspace`` for compressed model. "None" means random subspace.')

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
    parser.add_argument('--save_outs', action='store_true')

    # Save directory
    parser.add_argument('--parent_save_dir', type=str, default='./results/')

    args = parser.parse_args()

    return args


# Main function
def main():
    args = parse_args() # Get command line args
    print("args = ", args)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Input data
    data_type = args.data_type
    d = args.data_dim
    p = args.data_rank
    whiten_ = args.whiten
    normalize_ = args.normalize
    K = args.num_classes
    N_k = args.train_samples_per_class
    N = N_k * K
    N_k_val = args.val_samples_per_class
    data_dir = os.path.join(args.data_dir, data_type)


    # Network
    d_h = args.hidden_dim
    L = args.depth
    act_str = args.activation
    compress_ = args.compress
    rank = args.compress_rank if args.compress_rank is not None else K 
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

    save_outs = args.save_outs



    # Run training trials
    all_train_losses = np.zeros((num_trials, epochs))
    all_train_accs = np.zeros((num_trials, epochs))

    all_val_losses = np.zeros((num_trials, epochs))
    all_val_accs = np.zeros((num_trials, epochs))

    if save_outs:
        all_outs = np.zeros((num_trials, epochs, N, K))

    for i in tqdm.tqdm(range(num_trials)):
        print("===== TRIAL {} =====".format(i + 1))

        # Generate train samples
        if data_type == 'synthetic':
            train_set, train_loader = generate_gmm_data(d, K, N_k, data_rank=p, whiten_=whiten_, normalize_=normalize_, batch_size=batch_size)
            _, val_loader = generate_gmm_data(d, K, N_k_val, data_rank=p, whiten_=whiten_, normalize_=normalize_, batch_size=batch_size)
        elif data_type == 'fashion_mnist':
            train_set, train_loader = get_fashion_mnist(K, N_k, data_dir, batch_size=batch_size, train=True, shuffle=shuffle)
            _, val_loader = get_fashion_mnist(K, N_k_val, data_dir, batch_size=batch_size, train=False, shuffle=shuffle)
        else:
            raise ValueError("Invalid dataset.")
        
        # Loss function
        criterion = getattr(nn, loss_fn)(reduction='sum')

        # Train fully parameterized model
        model = MLP(L, d, d_h, K, act_str, init_method=init_method, init_scale=init_scale)
        if compress_: # Compress the model
            model = compress_model(model, rank, K, init_method, init_scale, criterion, train_set, subspace_type, device)

        # Print number of parameters
        num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print("Number of trainable parameters = ", num_params)

        # Train model
        optimizer = getattr(torch.optim, opt_type)(model.parameters(), lr=lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
        results = train(model, criterion, optimizer, train_loader, val_loader, epochs, device, scheduler=scheduler, return_outs=save_outs)

        # Store results
        all_train_losses[i, :] = results['train_losses']
        all_train_accs[i, :] = results['train_accs']

        all_val_losses[i, :] = results['val_losses']
        all_val_accs[i, :] = results['val_accs']

        if save_outs:
            all_outs[i, ...] = np.array(results['all_outs'])


    # Save results
    parent_save_dir = args.parent_save_dir
    model_exp_dir = act_str + '_network_L_' + str(L) + '_' + init_method + '_init_scale_' + str(init_scale) + '_eta_' + str(lr)
    data_exp_dir = data_type + '_d_' + str(d) + '_K_' + str(K) + '_N_' + str(N)
    
    save_dir = os.path.join(parent_save_dir, data_exp_dir, model_exp_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    root_fname = 'compressed_model_subspace_' + str(subspace_type) if compress_ else 'orig_model'
    np.savez(os.path.join(save_dir, root_fname + '_losses'), train_losses=all_train_losses, val_losses=all_val_losses)
    np.savez(os.path.join(save_dir, root_fname + '_accs'), train_accs=all_train_accs, val_accs=all_val_accs)
    if save_outs:
        np.savez(os.path.join(save_dir, root_fname + '_outs', outs=all_outs))


if __name__ == '__main__':
    main()

    


        