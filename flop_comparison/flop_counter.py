"""Analytic FLOP counting for three low-dimensional training schemes vs. a dense MLP.

Implements the "Required API" of flop_comparison/SPEC:

  A. Dense MLP baseline           (mlp/model.py:MLP, bias=False everywhere)
  B. Subspace-compressed scheme   (mlp/model.py:compress_model + CompressedMLP)
  C. Lottery Ticket Hypothesis    (Frankle & Carbin, ICLR 2019, Sec. 1/2: iterative
                                   magnitude pruning with weight rewinding)
  D. DLRT / low-rank lottery      (Schotthoefer et al., NeurIPS 2022, Algorithm 1
                                   rank-adaptive KLS integrator; cost model per Sec. 4.3)
"""

import math
from dataclasses import dataclass

MAC = 2  # FLOPs per multiply-accumulate


# ---------------------------------------------------------------------------
# Primitives (SPEC "Conventions")
# ---------------------------------------------------------------------------

def matmul_flops(m: int, k: int, n: int) -> int:
    """FLOPs of a dense (m x k) @ (k x n) matmul: 2*m*k*n (SPEC: MAC = 2)."""
    return MAC * m * k * n


def linear_fwd_flops(batch: int, in_dim: int, out_dim: int) -> int:
    """Forward GEMM of a bias-free Linear on a batch: 2*B*in*out."""
    return matmul_flops(batch, in_dim, out_dim)


def linear_bwd_flops(batch: int, in_dim: int, out_dim: int, *, need_dx: bool = True,
                     need_dw: bool = True) -> int:
    """Backward of a bias-free Linear: dx GEMM (2*B*in*out) if need_dx, plus
    dW GEMM (2*B*in*out) if need_dw (SPEC backward convention)."""
    total = 0
    if need_dx:
        total += matmul_flops(batch, in_dim, out_dim)
    if need_dw:
        total += matmul_flops(batch, in_dim, out_dim)
    return total


def qr_flops(m: int, n: int) -> int:
    """Householder QR with thin Q formed explicitly, dims sorted so m >= n:
    4*m*n^2 - (4/3)*n^3, as int."""
    if m < n:
        m, n = n, m
    return int(4 * m * n**2 - (4 / 3) * n**3)


def svd_flops(m: int, n: int) -> int:
    """Economy SVD with both singular-vector matrices (Golub-Van Loan, Golub-Reinsch),
    dims sorted so m >= n: 14*m*n^2 + 8*n^3, as int (SPEC `svd_flops`)."""
    if m < n:
        m, n = n, m
    return int(14 * m * n**2 + 8 * n**3)


# ---------------------------------------------------------------------------
# Dense MLP baseline
# ---------------------------------------------------------------------------

@dataclass
class MLPConfig:
    depth: int = 4          # L, number of Linear layers
    in_dim: int = 784       # d
    hidden_dim: int = 784   # d_h
    num_classes: int = 10   # K


def dense_layer_dims(cfg) -> list:
    """Weight shapes [(out, in), ...] of mlp/model.py:MLP:
    (d_h, d), (d_h, d_h) x (L-2), (K, d_h)."""
    dims = [(cfg.hidden_dim, cfg.in_dim)]
    dims += [(cfg.hidden_dim, cfg.hidden_dim)] * (cfg.depth - 2)
    dims.append((cfg.num_classes, cfg.hidden_dim))
    return dims


def dense_num_params(cfg) -> int:
    """Trainable parameter count of the dense MLP (1,851,808 for the canonical config)."""
    return sum(out * in_ for (out, in_) in dense_layer_dims(cfg))


def dense_fwd_flops(cfg, batch: int) -> int:
    """One dense forward pass on `batch` samples: sum of layer GEMMs."""
    return sum(linear_fwd_flops(batch, in_, out) for (out, in_) in dense_layer_dims(cfg))


def dense_bwd_flops(cfg, batch: int) -> int:
    """One dense backward pass: dW for every layer, dx for every layer except the
    first (input data requires no grad; SPEC backward convention)."""
    total = 0
    for i, (out, in_) in enumerate(dense_layer_dims(cfg)):
        total += linear_bwd_flops(batch, in_, out, need_dx=(i > 0), need_dw=True)
    return total


def dense_iter_flops(cfg, batch: int, opt_flops_per_param: int = 2) -> dict:
    """One dense training iteration: forward + backward + optimizer step."""
    fwd = dense_fwd_flops(cfg, batch)
    bwd = dense_bwd_flops(cfg, batch)
    opt = opt_flops_per_param * dense_num_params(cfg)
    return {'forward': fwd, 'backward': bwd, 'optimizer': opt,
            'total': fwd + bwd + opt}


# ---------------------------------------------------------------------------
# Subspace-compressed scheme
# ---------------------------------------------------------------------------
@dataclass
class SubspaceConfig:
    rank: int = 10               # r; subspace width R = 2*rank
    train_factors: bool = True   # True = U/V factors train (the repo's scheme); False = frozen-factor variant


def subspace_width(scfg) -> int:
    """R = 2*r, the fixed update-subspace width."""
    return 2 * scfg.rank


def subspace_num_params(cfg, scfg) -> int:
    """Trainable params of CompressedMLP: (L-1)*R^2 cores + d_h*K last layer,
    plus the R*(d + d_h) factor entries when train_factors (40,400 canonical)."""
    R = subspace_width(scfg)
    params = (cfg.depth - 1) * R * R + cfg.hidden_dim * cfg.num_classes
    if scfg.train_factors:
        params += R * (cfg.in_dim + cfg.hidden_dim)
    return params


def subspace_fwd_flops(cfg, scfg, batch: int) -> int:
    """CompressedMLP forward: V (d->R), (L-1) cores (R->R), U^T (R->d_h), W_L (d_h->K)
    = 2B[d*R + (L-1)*R^2 + R*d_h + d_h*K] (SPEC `subspace_fwd_flops`)."""
    R = subspace_width(scfg)
    total = linear_fwd_flops(batch, cfg.in_dim, R)                    # V
    total += (cfg.depth - 1) * linear_fwd_flops(batch, R, R)          # cores
    total += linear_fwd_flops(batch, R, cfg.hidden_dim)               # U^T
    total += linear_fwd_flops(batch, cfg.hidden_dim, cfg.num_classes)  # W_L
    return total


def subspace_bwd_flops(cfg, scfg, batch: int) -> int:
    """CompressedMLP backward (SPEC `subspace_bwd_flops`):
      - V: first layer -> never dx; dW iff train_factors;
      - cores: dW always; dx per the PyTorch rule (dx is skipped whenever the layer
        input requires no grad) -- with frozen factors (train_factors=False) the FIRST
        core's input is the output of a frozen V fed by data, so its dx GEMM is skipped,
        exactly as autograd/FlopCounterMode does;
      - U^T: dx always (grad must reach the cores); dW iff train_factors;
      - W_L: dx + dW.
    """
    R = subspace_width(scfg)
    total = linear_bwd_flops(batch, cfg.in_dim, R,
                             need_dx=False, need_dw=scfg.train_factors)      # V
    for i in range(cfg.depth - 1):                                           # cores
        need_dx = scfg.train_factors or i > 0
        total += linear_bwd_flops(batch, R, R, need_dx=need_dx, need_dw=True)
    total += linear_bwd_flops(batch, R, cfg.hidden_dim,
                              need_dx=True, need_dw=scfg.train_factors)      # U^T
    total += linear_bwd_flops(batch, cfg.hidden_dim, cfg.num_classes,
                              need_dx=True, need_dw=True)                    # W_L
    return total


def subspace_iter_flops(cfg, scfg, batch: int, opt_flops_per_param: int = 2) -> dict:
    """One compressed training iteration: forward + backward + optimizer step."""
    fwd = subspace_fwd_flops(cfg, scfg, batch)
    bwd = subspace_bwd_flops(cfg, scfg, batch)
    opt = opt_flops_per_param * subspace_num_params(cfg, scfg)
    return {'forward': fwd, 'backward': bwd, 'optimizer': opt,
            'total': fwd + bwd + opt}


def subspace_init_flops(cfg, scfg, num_train_samples: int, opt_flops_per_param: int = 2,
                        count_init_eval: bool = False) -> dict:
    """One-time cost of compress_model"""
    d, d_h, K, L = cfg.in_dim, cfg.hidden_dim, cfg.num_classes, cfg.depth
    R = subspace_width(scfg)
    N = num_train_samples

    probe_fwd = dense_fwd_flops(cfg, N)
    probe_bwd = dense_bwd_flops(cfg, N)
    probe_opt = opt_flops_per_param * dense_num_params(cfg)
    probe_eval = dense_fwd_flops(cfg, N) if count_init_eval else 0

    # M = vstack(G1, G1^T @ W1): one (d x d_h)(d_h x d) matmul.
    build = matmul_flops(d, d_h, d)

    sub_svd = svd_flops(d_h + d, d)
    factor_prop = matmul_flops(d_h, d, R) + (L - 2) * matmul_flops(d_h, d_h, R)

    out = {'probe_forward': probe_fwd,
           'probe_backward': probe_bwd,
           'probe_optimizer': probe_opt,
           'probe_eval': probe_eval,
           'subspace_build': build,
           'subspace_svd': sub_svd,
           'factor_propagation': factor_prop}
    out['total'] = sum(out.values())
    return out


# ---------------------------------------------------------------------------
# Lottery Ticket Hypothesis
# ---------------------------------------------------------------------------

@dataclass
class LTHConfig:
    prune_rate: float = 0.2      # fraction pruned per round (uniform across layers)
    n_rounds: int = 1
    iters_per_round: int = 500
    theoretical_sparsity: bool = True  # True: FLOPs scale with density; False: masked-dense


def lth_density(lcfg, after_round: int) -> float:
    """Surviving-weight fraction after `after_round` prune rounds: (1 - p)^after_round
    (LTH Sec. 1, iterative pruning: each round prunes p% of surviving weights)."""
    return (1 - lcfg.prune_rate) ** after_round


def lth_iter_flops(cfg, lcfg, batch: int, density: float,
                   opt_flops_per_param: int = 2) -> dict:
    """One training iteration of a ticket at the given density."""
    p_dense = dense_num_params(cfg)
    if lcfg.theoretical_sparsity:
        fwd = int(round(density * dense_fwd_flops(cfg, batch)))
        bwd = int(round(density * dense_bwd_flops(cfg, batch)))
        opt = opt_flops_per_param * math.ceil(density * p_dense)
        mask = 0
    else:
        fwd = dense_fwd_flops(cfg, batch)
        bwd = dense_bwd_flops(cfg, batch)
        opt = opt_flops_per_param * p_dense
        mask = p_dense
    return {'forward': fwd, 'backward': bwd, 'optimizer': opt,
            'mask_apply': mask, 'total': fwd + bwd + opt + mask}


def lth_search_flops(cfg, lcfg, batch: int, opt_flops_per_param: int = 2) -> dict:
    """Winning-ticket search = the LTH scheme's initialization cost"""
    training = 0
    for i in range(lcfg.n_rounds):
        density_i = lth_density(lcfg, i)
        training += lcfg.iters_per_round * \
            lth_iter_flops(cfg, lcfg, batch, density_i, opt_flops_per_param)['total']
    return {'training': training, 'total': training}


def lth_final_iter_flops(cfg, lcfg, batch: int, opt_flops_per_param: int = 2) -> dict:
    """Per-iteration cost of training the final winning ticket, at density (1-p)^n."""
    return lth_iter_flops(cfg, lcfg, batch, lth_density(lcfg, lcfg.n_rounds),
                          opt_flops_per_param)


def lth_rounds_for_density(prune_rate: float, target_density: float) -> int:
    """Smallest n >= 1 with (1 - p)^n <= target_density."""
    if not 0 < prune_rate < 1:
        raise ValueError(f"prune_rate must be in (0, 1), got {prune_rate}")
    if target_density <= 0:
        raise ValueError(f"target_density must be > 0, got {target_density}")
    n = 1
    while (1 - prune_rate) ** n > target_density:
        n += 1
    return n


# ---------------------------------------------------------------------------
# DLRT
# ---------------------------------------------------------------------------

@dataclass
class DLRTConfig:
    rank: int = 20               # fixed rank r per low-rank layer
    adaptive: bool = True        # rank-adaptive variant
    warm_start_svd: bool = False  # True: init factors by SVD of dense init weights

def _dlrt_low_rank_dims(cfg) -> list:
    """(out, in) of the low-rank layers: every Linear except the dense last layer."""
    return dense_layer_dims(cfg)[:-1]


def dlrt_num_params(cfg, dcfg) -> int:
    """Per low-rank layer W = U S V^T at rank r"""
    r = dcfg.rank
    total = sum(r * (out + in_) + r * r for (out, in_) in _dlrt_low_rank_dims(cfg))
    return total + cfg.hidden_dim * cfg.num_classes


def dlrt_init_flops(cfg, dcfg) -> dict:
    """DLRT initialization"""
    warm = 0
    if dcfg.warm_start_svd:
        warm = sum(svd_flops(out, in_) for (out, in_) in _dlrt_low_rank_dims(cfg))
    return {'warm_start': warm, 'total': warm}


def dlrt_iter_flops(cfg, dcfg, batch: int, opt_flops_per_param: int = 2) -> dict:
    """One iteration of DLRT Algorithm 1"""
    B = batch
    r = dcfg.rank
    R2 = 2 * r if dcfg.adaptive else r
    K_cls = cfg.num_classes
    d_h = cfg.hidden_dim
    dims = _dlrt_low_rank_dims(cfg)

    kl_form = kl_fwd = kl_bwd = qr = projection = s_fwd = s_bwd = svd_trunc = 0
    for i, (out, in_) in enumerate(dims):
        first = (i == 0)
        kl_form += 2 * out * r**2 + 2 * in_ * r**2
        # One factored forward serving both K and L tapes.
        kl_fwd += 2 * B * r * in_ + 2 * B * r * out
        # K/L backward: dz propagation (K^T dy then V(.)), dK = dy zhat^T, dL via U^T dy.
        if not first:
            kl_bwd += 2 * B * r * out + 2 * B * r * in_
        kl_bwd += 2 * B * out * r                       # dK
        kl_bwd += 2 * B * out * r + 2 * B * in_ * r     # dL
        qr += qr_flops(out, R2) + qr_flops(in_, R2)
        projection += 2 * out * R2 * r + 2 * in_ * R2 * r
        projection += 2 * R2 * r * r + 2 * R2 * r * R2
        # S-step forward at rank R2.
        s_fwd += 2 * B * R2 * in_ + 2 * B * R2**2 + 2 * B * R2 * out
        # S-step backward: w = U^T dy, dS = w zhat^T, dz propagation (skip first layer).
        s_bwd += 2 * B * out * R2 + 2 * B * R2**2
        if not first:
            s_bwd += 2 * B * R2**2 + 2 * B * in_ * R2
        if dcfg.adaptive:
            svd_trunc += svd_flops(R2, R2) + 2 * out * R2 * r + 2 * in_ * R2 * r

    # Dense last layer joins each pass: forward in both tapes, dx in the K/L backward,
    # dx + dW in the S-step backward (its dW is counted there).
    kl_fwd += 2 * B * d_h * K_cls
    kl_bwd += 2 * B * d_h * K_cls
    s_fwd += 2 * B * d_h * K_cls
    s_bwd += 4 * B * d_h * K_cls

    optimizer = opt_flops_per_param * (
        sum(out * r + in_ * r + R2**2 for (out, in_) in dims) + d_h * K_cls)

    out_dict = {'kl_form': kl_form, 'kl_fwd': kl_fwd, 'kl_bwd': kl_bwd, 'qr': qr,
                'projection': projection, 's_fwd': s_fwd, 's_bwd': s_bwd,
                'svd_truncate': svd_trunc, 'optimizer': optimizer}
    out_dict['total'] = sum(out_dict.values())
    return out_dict
