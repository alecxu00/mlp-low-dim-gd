import numpy as np
from typing import List, Union
import itertools
from scipy.linalg import sqrtm

    
def whiten(X: np.ndarray):
    ''' 
        X: np.array of shape (N, d)
    '''
    assert X.shape[0] >= X.shape[1]
    
    Sigma = X.T @ X
    P = sqrtm(np.linalg.inv(Sigma))
    
    X_whitened = X @ P

    return X_whitened


def normalize(X: np.ndarray):
    '''
        X: np array of shape (N, d)
    '''
    return X / np.linalg.norm(X, axis=1)


def svd_(X: np.ndarray, full_matrices: bool = True, compute_uv: bool = True):
    out = np.linalg.svd(X, compute_uv=compute_uv, full_matrices=full_matrices)
    if compute_uv:
        return out.U, out.S, out.Vh.T
    
    return out


def subspace_error(U_hat: np.ndarray, U: np.ndarray):
    return np.linalg.norm(U @ U.T - U_hat @ U_hat.T, ord='fro')**2 / np.linalg.norm(U @ U.T, ord='fro')**2


def compute_angles(U_1: np.ndarray, U_2: np.ndarray):
    return  np.arccos( np.clip( svd_(U_1.T @ U_2, compute_uv=False), 0, 1 ) ) #np.arccos(np.clip(np.abs(np.diag(U_1.T @ U_2)), 0, 1))

def complete_basis(U: np.array):
    d, r = U.shape
    
    U_orth = svd_(U, full_matrices=True)[0][:, r:]
    
    return np.concatenate((U_orth, U), axis=1)

def orthonormalize(X: np.array):
    if X.shape[0] >= X.shape[1]:
        return svd_(X, full_matrices=False)[0]
    else:
        return svd_(X, full_matrices=False)[-1].T


def get_weight_svds(W_epochs: List[List[np.ndarray]]):
    return [[svd_(W_l) for W_l in W_t] for W_t in W_epochs]


def left_right_inits(W_0: List[np.ndarray], G1_0: np.ndarray, r: int, eps: float, comp_type: int = 2, angle: float = 0.0):
    print("Computing initial U and V factors...")

    W1_0 = W_0[0]

    # Compute orthogonal complement of  N( G1(0) ) and N( G1^T(0) W1(0) ) )
    M = np.concatenate((G1_0, G1_0.T @ W1_0), axis=0) # Vertically stack G1(0) and G1^T(0) W1(0)

    # Get initialization of right factor
    right_all = svd_(M, full_matrices=False)[-1] # Initialization for right factor
    right_init = right_all[:, :2*r]
    if angle > 0: # Make it so the initialized subspace is some angle away from the "good" subspace
        right_init_orth = right_all[:, 2*r:4*r] # Get 2r-dim space that is orthogonal to "good" subspace
        right_init = right_init * np.cos(np.deg2rad(angle)) + right_init_orth * np.sin(np.deg2rad(angle))
    
    # Get initialization of left factor
    left_init = right_init.copy()

    if comp_type == 1:
        left_init = W1_0 @ left_init / eps
        
    elif comp_type == 2:
        for Wl_0 in W_0[:-1]:
            left_init = Wl_0 @ left_init / eps

    return left_init, right_init 

def left_right_inits_2(W_0: List[np.ndarray], E_0: np.ndarray, X: np.ndarray, r: int, eps: float, comp_type: int = 2, angle: float = 0.0):
    print("Computing initial U and V factors...")

    # Compute orthogonal complement of ( N( W_{L:2}^T(0) Gam(0) ) and N( Gam^T(0) W_{L:1} ) )
    Gam_0 = E_0 @ X.T 
    M1 = Gam_0.copy() # Shape (d, d)
    for Wl_0 in W_0[1:][::-1]:
        M1 = Wl_0.T @ M1

    M2 = Gam_0.copy().T
    for Wl_0 in W_0[::-1]:
        M2 = M2 @ Wl_0

    M = np.concatenate((M1, M2), axis=0) # Shape (2d, d)

    # Get initialization of right factor
    right_all = svd_(M, full_matrices=False)[-1] # Shape (d, d)
    right_init = right_all[:, :2*r] # Shape (d, 2r)
    if angle > 0: # Make it so the initialized subspace is some angle away from the "good" subspace
        right_init_orth = right_all[:, 2*r:4*r] # Get 2r-dim space that is orthogonal to "good" subspace
        right_init = right_init * np.cos(np.deg2rad(angle)) + right_init_orth * np.sin(np.deg2rad(angle))

    # Get initialization of left factor
    left_init = right_init.copy()

    if comp_type == 1:
        left_init = W_0[0] @ left_init / eps 

    elif comp_type == 2:
        for Wl_0 in W_0[:-1]:
            left_init = Wl_0 @ left_init / eps

    return left_init, right_init


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
    
    Gamma_0 = E_0 @ X.T
    M1 = Gamma_0.copy()
    for l in range(L - 1, 0, -1):
        Wl_0 = W_0[l]
        M1 = Wl_0.T @ M1
    
    M2 = Gamma_0.copy().T
    for l in range(L - 1, -1, -1):
        Wl_0 = W_0[l]
        M2 = M2 @ Wl_0
    
    M1_V_r = svd_(M1)[-1][:, :r]
    M2_V_r = svd_(M2)[-1][:, :r]
    
    V_tilde = orthcomp_intersection(M1_V_r, M2_V_r)
    U_tilde = W_0[0] @ V_tilde / eps
    
    return U_tilde, V_tilde




def save_weight_svd_errors(W_epochs: List[List[np.ndarray]], 
                           G_epochs: List[List[np.ndarray]], 
                           E_0: np.ndarray, 
                           X: np.ndarray, 
                           r: int, 
                           eps: float,
                           subspace_type: int = 1):
    epochs = len(W_epochs) - 1
    UsV_epochs = get_weight_svds(W_epochs)
    UsV_0 = UsV_epochs[0]

    L = len(UsV_0)
    d = UsV_0[0][0].shape[0]
    
    U_top_errors = np.zeros((epochs, L - 1))
    U_mid_errors = np.zeros((epochs, L - 1))
    U_bot_errors = np.zeros((epochs, L - 1))

    V_top_errors = np.zeros((epochs, L - 1))
    V_mid_errors = np.zeros((epochs, L - 1))
    V_bot_errors = np.zeros((epochs, L - 1))

    svals = np.zeros((epochs, L - 1, d))
    
    
    for t in range(1, epochs + 1):
        #print("====== Saving Epoch", t, "======")
        UsV_t = UsV_epochs[t]
        UsV_prev = UsV_epochs[t - 1]
        
        if subspace_type == 1:
            U_null, V_null = small_update_subspace(W_epochs[0][0], G_epochs[0][0], r, eps)
        elif subspace_type == 2:
            U_null, V_null = small_update_subspace_2(W_epochs[0], E_0, X, r, eps)
        else:
            raise ValueError('Invalid subspace type')
            
        for l in range(L - 1):
            U0, _, V0 = UsV_0[l]
            Uprev, _, Vprev = UsV_prev[l]
            Ut, st, Vt = UsV_t[l]
           
            svals[t - 1, l, :] = st

            # U errors
            Ut_top = Ut[:, :r]
            Ut_mid = Ut[:, r:-r]
            Ut_bot = Ut[:, -r:]
            
            Uprev_top = Uprev[:, :r]
            Uprev_mid = Uprev[:, r:-r]
            Uprev_bot = Uprev[:, -r:]

            U0_top = U0[:, :r]
            U0_mid = U0[:, r:-r]
            U0_bot = U0[:, -r:]

            Ut_top_error = subspace_error(Ut_top, U0_top)
            Ut_mid_error = subspace_error(Ut_mid, U_null) 
            Ut_bot_error = subspace_error(Ut_bot, U0_bot)

            U_top_errors[t - 1, l] = Ut_top_error
            U_mid_errors[t - 1, l] = Ut_mid_error
            U_bot_errors[t - 1, l] = Ut_bot_error


            # V errors
            Vt_top = Vt[:, :r]
            Vt_mid = Vt[:, r:-r]
            Vt_bot = Vt[:, -r:]
            
            Vprev_top = Vprev[:, :r]
            Vprev_mid = Vprev[:, r:-r]
            Vprev_bot = Vprev[:, -r:]

            V0_top = V0[:, :r]
            V0_mid = V0[:, r:-r]
            V0_bot = V0[:, -r:]

            Vt_top_error = subspace_error(Vt_top, V0_top)
            Vt_mid_error = subspace_error(Vt_mid, V_null)
            Vt_bot_error = subspace_error(Vt_bot, V0_bot)

            V_top_errors[t - 1, l] = Vt_top_error
            V_mid_errors[t - 1, l] = Vt_mid_error
            V_bot_errors[t - 1, l] = Vt_bot_error
            
            # Update subspaces
            V_null = U_null
            U_null = W_epochs[0][l + 1] @ V_null / eps
        

    result = {'U_top_errors': U_top_errors,
              'U_mid_errors': U_mid_errors,
              'U_bot_errors': U_bot_errors,
              'V_top_errors': V_top_errors,
              'V_mid_errors': V_mid_errors,
              'V_bot_errors': V_bot_errors,
              'svals': svals}
    
    return result


def save_weight_svd_angles(W_epochs: List[List[np.ndarray]], G_epochs: List[List[np.ndarray]], r: int, eps: float):
    epochs = len(W_epochs) - 1
    UsV_epochs = get_weight_svds(W_epochs)
    UsV_0 = UsV_epochs[0]

    L = len(UsV_0)
    d = UsV_0[0][0].shape[0]
    
    U_top_angles = np.zeros((epochs, L - 1, r))
    U_mid_angles = np.zeros((epochs, L - 1, d - 2*r))
    U_bot_angles = np.zeros((epochs, L - 1, r))

    V_top_angles = np.zeros((epochs, L - 1, r))
    V_mid_angles = np.zeros((epochs, L - 1, d - 2*r))
    V_bot_angles = np.zeros((epochs, L - 1, r))

    svals = np.zeros((epochs, L - 1, d))
    
    
    for t in range(1, epochs + 1):
        #print("====== Saving Epoch", t, "======")
        UsV_t = UsV_epochs[t]
        
        U_null, V_null = small_update_subspace(W_epochs[0][0], G_epochs[0][0], r, eps)
        for l in range(L - 1):
            U_0, _, V_0 = UsV_0[l]
            U_t, s_t, V_t = UsV_t[l]
            svals[t - 1, l, :] = s_t
            

            # U angles
            Ut_top = U_t[:, :r]
            Ut_mid = U_t[:, r:-r]
            Ut_bot = U_t[:, -r:]

            U0_top = U_0[:, :r]
            U0_mid = U_0[:, r:-r]
            U0_bot = U_0[:, -r:]

            Ut_top_angles = compute_angles(Ut_top, U0_top)
            Ut_mid_angles = compute_angles(Ut_mid, U_null)
            Ut_bot_angles = compute_angles(Ut_bot, U0_bot)

            U_top_angles[t - 1, l] = Ut_top_angles
            U_mid_angles[t - 1, l] = Ut_mid_angles
            U_bot_angles[t - 1, l] = Ut_bot_angles


            # V angles
            Vt_top = V_t[:, :r]
            Vt_mid = V_t[:, r:-r]
            Vt_bot = V_t[:, -r:]

            V0_top = V_0[:, :r]
            V0_mid = V_0[:, r:-r]
            V0_bot = V_0[:, -r:]

            Vt_top_angles = compute_angles(Vt_top, V0_top)
            Vt_mid_angles = compute_angles(Vt_mid, V_null)
            Vt_bot_angles = compute_angles(Vt_bot, V0_bot)

            V_top_angles[t - 1, l, :] = Vt_top_angles
            V_mid_angles[t - 1, l, :] = Vt_mid_angles
            V_bot_angles[t - 1, l, :] = Vt_bot_angles
            
            # Update subspaces
            V_null = U_null
            U_null = W_epochs[0][l + 1] @ V_null / eps

    U_all_angles = np.concatenate((U_top_angles, U_mid_angles, U_bot_angles), axis=-1)
    V_all_angles = np.concatenate((V_top_angles, V_mid_angles, V_bot_angles), axis=-1)

    # result = {'U_top_angles': U_top_angles,
    #           'U_mid_angles': U_mid_angles,
    #           'U_bot_angles': U_bot_angles,
    #           'V_top_angles': V_top_angles,
    #           'V_mid_angles': V_mid_angles,
    #           'V_bot_angles': V_bot_angles,
    #           'svals': svals}
    
    result = {'U_all_angles': U_all_angles,
              'V_all_angles': V_all_angles,
              'svals': svals}
    
    return result


