import numpy as np
from typing import List
from scipy.linalg import sqrtm

    
def whiten(X: np.ndarray):
    ''' 
        Makes the features orthogonal
        X: np.array of shape (N, d)
    '''
    assert X.shape[0] >= X.shape[1]
    
    Sigma = X.T @ X
    P = sqrtm(np.linalg.inv(Sigma))
    
    X_whitened = X @ P

    return X_whitened


def orthogonalize(X: np.ndarray):
    ''' 
        Makes the data samples orthogonal
        X: np.array of shape (N, d)
    '''
    
    assert X.shape[0] <= X.shape[1]

    Sigma = X @ X.T
    P = sqrtm(np.linalg.inv(Sigma))

    X_orth = P @ X

    return X_orth


def normalize(X: np.ndarray):
    '''
        X: np array of shape (N, d)
    '''
    return X / np.expand_dims( np.linalg.norm(X, axis=1), axis=-1 )


def svd_(X: np.ndarray, full_matrices: bool = True, compute_uv: bool = True):
    out = np.linalg.svd(X, compute_uv=compute_uv, full_matrices=full_matrices)
    if compute_uv:
        return out.U, out.S, out.Vh.T
    
    return out


def subspace_error(U_hat: np.ndarray, U: np.ndarray):
    return np.linalg.norm(U @ U.T - U_hat @ U_hat.T, ord='fro')**2 / ( 2 * np.linalg.norm(U @ U.T, ord='fro')**2 )


def get_weight_svds(W_epochs: List[List[np.ndarray]]):
    return [[svd_(W_l) for W_l in W_t] for W_t in W_epochs]


def left_right_inits(W_0: List[np.ndarray], G1_0: np.ndarray, r: int, eps: float, angle: float = 0.0):
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
    L, s, R = svd_(G1_0, full_matrices=False)
    
    orthcomp_null_G1_0 = R[:, :r] #svd_(G1_0)[-1][:, :r] # Orthogonal complement of "null space" of G1_0 (U1)
    #orthcomp_null_G1_W1_0 = svd_(G1_0.T @ W1_0 / eps)[-1][:, :r] # Orthogonal complement of "null space" of G1_0.T @ W1_0 (U2)
    
    M = W1_0.T @ L[:, :r] / eps
    orthcomp_null_G1_W1_0 = svd_(M)[0][:, :r]
    
    
    # Get basis for orthogonal complement of (U1 + U2) ==> (U1_perp) and (U2_perp)
    V_null = orthcomp_intersection(orthcomp_null_G1_0, orthcomp_null_G1_W1_0) 
    U_null = W1_0 @ V_null / eps
    
    return U_null, V_null


def save_weight_svd_errors(W_epochs: List[List[np.ndarray]], 
                           G1_0: np.ndarray, 
                           r: int, 
                           eps: float):
    
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
        
        U_null, V_null = small_update_subspace(W_epochs[0][0], G1_0, r, eps)
            
        for l in range(L - 1):
            U0, _, V0 = UsV_0[l]
            Ut, st, Vt = UsV_t[l]
           
            svals[t - 1, l, :] = st

            # U errors
            Ut_top = Ut[:, :r]
            Ut_mid = Ut[:, r:-r]
            Ut_bot = Ut[:, -r:]
            
            U0_top = U0[:, :r]
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
            
            V0_top = V0[:, :r]
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
