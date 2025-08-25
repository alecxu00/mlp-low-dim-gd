import numpy as np
from typing import List, Union
import itertools
from scipy.linalg import sqrtm

def svd_(X: np.ndarray, full_matrices: bool = True, compute_uv: bool = True):
    out = np.linalg.svd(X, compute_uv=compute_uv, full_matrices=full_matrices)
    if compute_uv:
        return out.U, out.S, out.Vh.T
    
    return out


def complete_basis(U: np.array):
    _, r = U.shape
    U_orth = svd_(U, full_matrices=True)[0][:, r:]
    
    return np.concatenate((U_orth, U), axis=1)


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

def left_right_inits_2(W_0: List[np.ndarray], E_0: np.ndarray, X: np.ndarray, r: int, eps: float):
    print("Computing initial U and V factors...")

    # Compute orthogonal complement of ( N( W_{L:2}^T(0) Gam(0) ) and N( Gam^T(0) W_{L:1} ) )
    Gam_0 = E_0 @ X.T
    M1 = Gam_0.copy()
    for Wl_0 in W_0[1:][::-1]:
        M1 = Wl_0.T @ M1

    M2 = Gam_0.copy().T
    for Wl_0 in W_0[::-1]:
        M2 = M2 @ Wl_0

    M = np.concatenate((M1, M2), axis=0)

    # Get initializations
    right_ = svd_(M, full_matrices=False)[-1][:, :2*r]

    left_ = right_.copy()
    for Wl_0 in W_0[:-1]:
        left_ = Wl_0 @ left_ / eps

    return left_, right_



def orthcomp_intersection(U1: np.ndarray, U2: np.ndarray):
    d, r = U1.shape
    assert U2.shape == (d, r) and r <= (d / 2)
    
    Q = svd_( np.concatenate((U1, U2), axis=1) )[0] 
    orthcomp_intersec = Q[:, 2*r:] # Basis for ortho complement of (U1 + U2) ==> basis for U1_perp and U2_perp
    
    return orthcomp_intersec


def small_update_subspace(W1_0: np.ndarray, G1_0: np.ndarray, r: int, eps: float):    
    orthcomp_null_G1_0 = svd_(G1_0)[-1][:, :r] # Orthogonal complement of "null space" of G1_0 (U1)
    orthcomp_null_G1_W1_0 = svd_(G1_0.T @ W1_0)[-1][:, :r] # Orthogonal complement of "null space" of G1_0.T @ W1_0 (U2)
    
    V_null = orthcomp_intersection(orthcomp_null_G1_0, orthcomp_null_G1_W1_0) # Get basis for orthogonal complement of (U1 + U2) ==> (U1_perp) and (U2_perp) ==> N(G1_0) and N(G1_0.T @ W1_0)
    U_null = W1_0 @ V_null / eps
    
    return U_null, V_null


def small_update_subspace_2(W_0: List[np.ndarray], E_0: np.ndarray, X: np.ndarray, r: int, eps: float):
    L = len(W_0)
    
    Gam_0 = E_0 @ X.T
    M1 = Gam_0.copy()
    for Wl_0 in W_0[1:][::-1]:
        M1 = Wl_0.T @ M1

    M2 = Gam_0.copy().T
    for Wl_0 in W_0[::-1]:
        M2 = M2 @ Wl_0
    
    M1_V_r = svd_(M1)[-1][:, :r]
    M2_V_r = svd_(M2)[-1][:, :r]
    
    V_tilde = orthcomp_intersection(M1_V_r, M2_V_r)
    U_tilde = W_0[0] @ V_tilde / eps
    
    return U_tilde, V_tilde