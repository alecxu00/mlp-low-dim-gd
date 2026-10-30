import numpy as np
from typing import List

def svd_(X: np.ndarray, full_matrices: bool = True, compute_uv: bool = True):
    out = np.linalg.svd(X, compute_uv=compute_uv, full_matrices=full_matrices)
    if compute_uv:
        return out.U, out.S, out.Vh.T
    
    return out


def left_right_inits(W_0: List[np.ndarray], G1_0: np.ndarray, r: int, eps: float):
    print("Computing initial U and V factors...")

    W1_0 = W_0[0]

    # Compute orthogonal complement of  N( G1(0) ) and N( G1^T(0) W1(0) ) )
    M = np.concatenate((G1_0, G1_0.T @ W1_0), axis=0) # Vertically stack G1(0) and G1^T(0) W1(0)
    right_ = svd_(M, full_matrices=False)[-1][:, :2*r] # Initialization for right factor

    left_ = right_.copy()
    for Wl_0 in W_0[:-1]:
        left_ = Wl_0 @ left_ / eps

    return left_, right_ 
