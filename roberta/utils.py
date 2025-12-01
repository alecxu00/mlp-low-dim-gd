from typing import List
import numpy as np

def left_right_inits(W_0: List[np.ndarray],
                     G1_0: np.ndarray, 
                     rank: int, 
                     init_scale: float = 1.0, 
                     angle: float = 0.0):
    
    print("Computing initial U and V factors...")

    # Number of layers and first-layer weights at initialization
    W1_0 = W_0[0]

    # Compute orthogonal complement of  N( G1(0) ) and N( G1^T(0) W1(0) ) )
    M = np.concatenate((G1_0, G1_0.T @ (W1_0 / init_scale)), axis=0) # Vertically stack G1(0) and G1^T(0) W1(0)

    # Get initialization of right factor
    right_all = np.linalg.svd(M, full_matrices=False).Vh.T # Initialization for right factor
    right_init = right_all[:, :rank]
    if angle > 0: # Make it so the initialized subspace is some angle away from the "good" subspace
        right_init_orth = right_all[:, rank:2*rank] # Get 2r-dim space that is orthogonal to "good" subspace
        right_init = right_init * np.cos(np.deg2rad(angle)) + right_init_orth * np.sin(np.deg2rad(angle))
    
    # Get initialization of left factor
    left_init = right_init.copy()

    for Wl_0 in W_0[:-1]:
        left_init = Wl_0 @ left_init / init_scale

    return left_init, right_init 