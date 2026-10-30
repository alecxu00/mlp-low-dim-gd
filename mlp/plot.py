import numpy as np
import matplotlib.pyplot as plt
# LineCollection, PolyCollection, MaxNLocator, MultipleLocator and Dict are only used by main_fig.ipynb (via `from plot import *`)
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import MaxNLocator, MultipleLocator
from typing import Dict, List, Union


def plot_train_losses(train_losses: Union[List, np.array],
                      act_fn_str: str, 
                      d: int, 
                      K: int, 
                      save_fname: str, 
                      savefig: bool = False):
    
    
    title_ = act_fn_str + ' Network Train Loss (d = ' + str(d) + ', K = ' + str(K) + ')'
    plt.figure()
    if isinstance(train_losses, np.ndarray) and len(train_losses.shape) > 1:
        mean_train_losses = np.mean(train_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        
        plt.plot(np.arange(len(mean_train_losses)) + 1, mean_train_losses)
        plt.fill_between(np.arange(len(mean_train_losses)) + 1, mean_train_losses - std_train_losses, mean_train_losses+ std_train_losses, alpha=0.5)
    
    else:
        plt.plot(np.arange(len(train_losses)) + 1, train_losses)
    
    plt.title(title_)
    plt.xlabel('Iteration (t)')
    plt.ylabel('Train Loss')
    
    if savefig:
        plt.savefig(save_fname)
    plt.show()
    plt.close()
