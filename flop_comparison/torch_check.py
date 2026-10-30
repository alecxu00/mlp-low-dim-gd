"""
    Empirical validation of flop_counter.py against torch.utils.flop_counter.FlopCounterMode.
"""

import argparse
import logging
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, 'mlp'))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.flop_counter import FlopCounterMode

import flop_counter as fc
from model import MLP, CompressedMLP, compress_model  # real repo classes (mlp/model.py)
from data import generate_gmm_data                     # real repo data (mlp/data.py)

logging.disable(logging.INFO)  # silence the 1-epoch probe train() inside compress_model


class ProbeDataset(Dataset):
    """Synthetic-data wrapper for compress_model"""

    def __init__(self, samples: np.ndarray, targets: np.ndarray):
        self.samples = samples
        self.data = samples
        self.targets = targets

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class KStepNet(nn.Module):
    """DLRT Alg. 1 K-step parameterization (SPEC torch_check item 4): per low-rank
    layer a frozen V Linear(in -> r) followed by a trainable K = U S Linear(r -> out),
    plus the dense last layer."""

    def __init__(self, low_rank_dims, rank: int, hidden_dim: int, num_classes: int):
        super().__init__()
        layers = []
        for (out, in_) in low_rank_dims:
            v = nn.Linear(in_, rank, bias=False)
            v.weight.requires_grad_(False)
            layers += [v, nn.Linear(rank, out, bias=False), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, num_classes, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_forward(model, x) -> int:
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            model(x)
    return counter.get_total_flops()


def count_forward_backward(model, x) -> int:
    for p in model.parameters():
        p.grad = None
    with FlopCounterMode(display=False) as counter:
        out = model(x)
        if isinstance(out, tuple):  # MLP / CompressedMLP return (out, hidden_features)
            out = out[0]
        out.sum().backward()  # sum/loss are elementwise: not counted anyway
    return counter.get_total_flops()


def set_factor_requires_grad(comp_model, flag: bool) -> int:
    """Toggle requires_grad on the two subspace factor layers (U, V) of CompressedMLP."""
    n = 0
    for entry in comp_model.subspace_params:
        if isinstance(entry, nn.Linear):
            entry.weight.requires_grad_(flag)
        else:  # stored directly as a Parameter
            entry.requires_grad_(flag)
        n += 1
    return n


def time_train_iter(model, x, y, criterion, device, iters=20, warmup=3) -> float:
    """Median wall-clock seconds of one training iteration (fwd + bwd + SGD step)."""
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    times = []
    for i in range(warmup + iters):
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad()
        out, _ = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        if device == 'cuda':
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--small', action='store_true',
                        help='quick CPU config (L=4, d=16, d_h=16, K=4, r=2, B=8) '
                             'instead of L=3, d=784, d_h=784, K=10, r=10, B=256')
    args = parser.parse_args(argv)

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            print('NOTE: --device auto found no GPU, falling back to cpu '
                  '(FlopCounterMode counts are device-independent).')
    else:
        device = args.device

    if args.small:
        L, d, d_h, K, r, B, N = 4, 16, 16, 4, 2, 8, 32
    else:
        L, d, d_h, K, r, B, N = 3, 784, 784, 10, 10, 256, 256

    torch.manual_seed(0)
    np.random.seed(0)

    cfg = fc.MLPConfig(depth=L, in_dim=d, hidden_dim=d_h, num_classes=K)
    scfg_impl = fc.SubspaceConfig(rank=r, train_factors=True)
    scfg_frozen = fc.SubspaceConfig(rank=r, train_factors=False)
    print(f'Config: L={L} d={d} d_h={d_h} K={K} r={r} B={B} N={N} device={device}\n')

    # Synthetic data through the real repo generator (returns 3 values).
    dataset, _, _ = generate_gmm_data(d=d, K=K, N_k=math.ceil(N / K))
    probe_set = ProbeDataset(dataset.samples, dataset.targets)
    x = torch.tensor(dataset.samples[:B], dtype=torch.float32, device=device)
    y = torch.tensor(dataset.targets[:B], dtype=torch.float32, device=device)
    criterion = nn.MSELoss(reduction='sum')

    checks = []  # (name, analytic, measured)

    # 1. Dense MLP forward / forward+backward.
    dense = MLP(num_layers=L, in_dim=d, hidden_dim=d_h, num_classes=K,
                activation='GELU', init_method='orthogonal', init_scale=0.1).to(device)
    dense_fwd = fc.dense_fwd_flops(cfg, B)
    checks.append(('dense forward', dense_fwd, count_forward(dense, x)))
    checks.append(('dense forward+backward', dense_fwd + fc.dense_bwd_flops(cfg, B),
                   count_forward_backward(dense, x)))

    # 2. Real CompressedMLP via compress_model (1-epoch full-batch probe inside).
    orig = MLP(num_layers=L, in_dim=d, hidden_dim=d_h, num_classes=K,
               activation='GELU', init_method='orthogonal', init_scale=0.1)
    comp = compress_model(orig, rank=r, num_classes=K, init_method='orthogonal',
                          init_scale=0.1, criterion=criterion, train_set=probe_set,
                          subspace_type='gradient', device=device, angle=0.0)
    assert isinstance(comp, CompressedMLP)
    comp = comp.to(device)
    n_trainable = sum(p.numel() for p in comp.parameters() if p.requires_grad)
    checks.append(('compressed trainable params',
                   fc.subspace_num_params(cfg, scfg_impl), n_trainable))
    sub_fwd = fc.subspace_fwd_flops(cfg, scfg_impl, B)
    checks.append(('compressed forward', sub_fwd, count_forward(comp, x)))
    checks.append(('compressed fwd+bwd (train_factors=True)',
                   sub_fwd + fc.subspace_bwd_flops(cfg, scfg_impl, B),
                   count_forward_backward(comp, x)))

    # 3. Frozen-factor variant: freeze U and V, re-count the backward.
    n_factors = set_factor_requires_grad(comp, False)
    assert n_factors == 2, f'expected 2 subspace factor layers, found {n_factors}'
    checks.append(('compressed fwd+bwd (train_factors=False)',
                   sub_fwd + fc.subspace_bwd_flops(cfg, scfg_frozen, B),
                   count_forward_backward(comp, x)))
    set_factor_requires_grad(comp, True)  # restore for timing

    # 4. DLRT component check: K-step-parameterization forward vs 'kl_fwd'.
    cfg3 = fc.MLPConfig(depth=3, in_dim=d, hidden_dim=d_h, num_classes=K)
    low_rank_dims = fc.dense_layer_dims(cfg3)[:-1]
    kstep = KStepNet(low_rank_dims, r, d_h, K).to(device)
    kl_fwd_primitives = sum(fc.linear_fwd_flops(B, in_, r) + fc.linear_fwd_flops(B, r, out)
                            for (out, in_) in low_rank_dims)
    kl_fwd_primitives += fc.linear_fwd_flops(B, d_h, K)
    assert kl_fwd_primitives == fc.dlrt_iter_flops(cfg3, fc.DLRTConfig(rank=r), B)['kl_fwd']
    checks.append(('DLRT K-step forward (kl_fwd)', kl_fwd_primitives,
                   count_forward(kstep, x)))

    # Report.
    name_w = max(len(n) for n, _, _ in checks)
    print(f'{"check".ljust(name_w)}  {"analytic":>16}  {"measured":>16}  match')
    print('-' * (name_w + 45))
    ok = True
    for name, analytic, measured in checks:
        match = analytic == measured
        ok &= match
        print(f'{name.ljust(name_w)}  {analytic:>16,}  {measured:>16,}  '
              f'{"OK" if match else "MISMATCH"}')

    # Wall-clock timing of one training iteration, dense vs compressed.
    label = 'small config' if args.small else 'table config'
    if args.small:
        print(f'\nTiming at the {label} (--small given; table-config timing is the '
              'default):')
    else:
        print(f'\nTiming at the {label}:')
    t_dense = time_train_iter(dense, x, y, criterion, device)
    t_comp = time_train_iter(comp, x, y, criterion, device)
    print(f'  dense      median iter: {t_dense * 1e3:9.3f} ms')
    print(f'  compressed median iter: {t_comp * 1e3:9.3f} ms  '
          f'(speedup {t_dense / t_comp:.2f}x)')

    print(f'\n{"ALL COMPARISONS MATCH EXACTLY" if ok else "MISMATCHES FOUND"}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
