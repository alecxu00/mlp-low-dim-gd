"""pytest suite for flop_comparison/ (SPEC "tests/" section, coverage items 1-8).

All primitive / dense / subspace / hand-built-sum / FlopCounterMode-oracle checks are
EXACT integer equalities (==, no tolerances). Oracle tests run on CPU with the small
config L=4, d=d_h=16, K=4, r=2, B=8, N=32 and are seeded for determinism.
"""

import json
import math

import numpy as np
import pytest
import torch
import torch.nn as nn

import compare
import flop_counter as fc
import torch_check as tc  # import-safe harness: ProbeDataset, KStepNet, count_* helpers

from model import MLP, CompressedMLP, compress_model  # real repo classes (mlp/)
from data import generate_gmm_data                     # real repo data (mlp/)

# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------

CANON_CFG = fc.MLPConfig()   # L=4, d=784, d_h=784, K=10
CANON_N = 60000              # full-batch GD: B = N
CANON_B = 60000

# Small oracle config: L=4, d=d_h=16, K=4, r=2 (R=4), B=8, N=32.
S_L, S_D, S_DH, S_K, S_R, S_B, S_N = 4, 16, 16, 4, 2, 8, 32
SMALL_CFG = fc.MLPConfig(depth=S_L, in_dim=S_D, hidden_dim=S_DH, num_classes=S_K)
SMALL_SCFG = fc.SubspaceConfig(rank=S_R, train_factors=True)
SMALL_SCFG_FROZEN = fc.SubspaceConfig(rank=S_R, train_factors=False)

# Hand-computed integers for the small config (2*B*sum of GEMM dims):
DENSE_FWD_SMALL = 13312       # 2*8*(3*16*16 + 16*4)
DENSE_BWD_SMALL = 22528       # dW all layers + dx all but first
SUB_FWD_SMALL = 3840          # 2*8*(16*4 + 3*4*4 + 4*16 + 16*4)
SUB_BWD_TRAIN_SMALL = 6656    # V dW; 3 cores dx+dW; U^T dx+dW; W_L dx+dW
SUB_BWD_FROZEN_SMALL = 4352   # V none; core0 dW only; cores 1-2 dx+dW; U^T dx; W_L dx+dW
KL_FWD_SMALL = 3072           # depth-3 K-step net: 2*(512+512) + 1024


# ---------------------------------------------------------------------------
# 1. Primitives, exact (SPEC coverage item 1)
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_mac_constant(self):
        assert fc.MAC == 2

    def test_matmul_flops(self):
        assert fc.matmul_flops(1, 1, 1) == 2
        assert fc.matmul_flops(3, 4, 5) == 120
        assert fc.matmul_flops(784, 784, 784) == 2 * 784 ** 3

    def test_linear_fwd_flops(self):
        assert fc.linear_fwd_flops(8, 16, 32) == 8192
        assert fc.linear_fwd_flops(1, 784, 10) == 15680

    def test_linear_bwd_flops_combinations(self):
        # dx GEMM and dW GEMM are each 2*B*in*out.
        assert fc.linear_bwd_flops(8, 16, 32) == 16384
        assert fc.linear_bwd_flops(8, 16, 32, need_dx=True, need_dw=False) == 8192
        assert fc.linear_bwd_flops(8, 16, 32, need_dx=False, need_dw=True) == 8192
        assert fc.linear_bwd_flops(8, 16, 32, need_dx=False, need_dw=False) == 0

    def test_svd_flops(self):
        # 14*m*n^2 + 8*n^3 with m >= n.
        assert fc.svd_flops(10, 4) == 2752
        assert fc.svd_flops(4, 10) == 2752   # m < n swap
        assert fc.svd_flops(5, 5) == 2750


# ---------------------------------------------------------------------------
# Dataclass defaults per SPEC "Required API"
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    def test_mlp_config_defaults(self):
        cfg = fc.MLPConfig()
        assert (cfg.depth, cfg.in_dim, cfg.hidden_dim, cfg.num_classes) == (4, 784, 784, 10)

    def test_subspace_config_defaults(self):
        scfg = fc.SubspaceConfig()
        assert (scfg.rank, scfg.train_factors) == (10, True)

    def test_lth_config_defaults(self):
        lcfg = fc.LTHConfig()
        assert (lcfg.prune_rate, lcfg.n_rounds, lcfg.iters_per_round,
                lcfg.theoretical_sparsity) == (0.2, 1, 500, True)

    def test_dlrt_config_defaults(self):
        dcfg = fc.DLRTConfig()
        assert (dcfg.rank, dcfg.adaptive, dcfg.warm_start_svd) == (20, True, False)


# ---------------------------------------------------------------------------
# 2. Dense analytic, exact (SPEC coverage item 2)
# ---------------------------------------------------------------------------

class TestDenseAnalytic:
    def test_layer_dims_canonical(self):
        assert fc.dense_layer_dims(CANON_CFG) == [(784, 784), (784, 784),
                                                  (784, 784), (10, 784)]

    def test_layer_dims_asymmetric(self):
        cfg = fc.MLPConfig(depth=3, in_dim=4, hidden_dim=5, num_classes=2)
        assert fc.dense_layer_dims(cfg) == [(5, 4), (5, 5), (2, 5)]

    def test_num_params_canonical_regression(self):
        # Verified against mlp/logs: 3*784^2 + 10*784.
        assert fc.dense_num_params(CANON_CFG) == 1_851_808

    def test_num_params_small(self):
        assert fc.dense_num_params(SMALL_CFG) == 832  # 3*256 + 64

    def test_fwd_bwd_hand_config(self):
        # L=3, d=4, d_h=5, K=2, B=7: fwd = 14*(20+25+10) = 770;
        # bwd = 280 (dW only, first) + 700 (dx+dW) + 280 (dx+dW) = 1260.
        cfg = fc.MLPConfig(depth=3, in_dim=4, hidden_dim=5, num_classes=2)
        assert fc.dense_fwd_flops(cfg, 7) == 770
        assert fc.dense_bwd_flops(cfg, 7) == 1260

    def test_fwd_bwd_small_config(self):
        assert fc.dense_fwd_flops(SMALL_CFG, S_B) == DENSE_FWD_SMALL
        assert fc.dense_bwd_flops(SMALL_CFG, S_B) == DENSE_BWD_SMALL

    def test_bwd_primitive_built(self):
        # First layer: dW only; all other layers: dx + dW.
        cfg = fc.MLPConfig(depth=4, in_dim=6, hidden_dim=10, num_classes=3)
        B = 5
        expected = (fc.linear_bwd_flops(B, 6, 10, need_dx=False, need_dw=True)
                    + 2 * fc.linear_bwd_flops(B, 10, 10)
                    + fc.linear_bwd_flops(B, 10, 3))
        assert fc.dense_bwd_flops(cfg, B) == expected

    def test_iter_dict(self):
        d = fc.dense_iter_flops(SMALL_CFG, S_B)
        assert set(d) == {'forward', 'backward', 'optimizer', 'total'}
        assert d['forward'] == DENSE_FWD_SMALL
        assert d['backward'] == DENSE_BWD_SMALL
        assert d['optimizer'] == 2 * fc.dense_num_params(SMALL_CFG)
        assert d['total'] == d['forward'] + d['backward'] + d['optimizer']

    def test_iter_dict_adam(self):
        d = fc.dense_iter_flops(SMALL_CFG, S_B, opt_flops_per_param=12)
        assert d['optimizer'] == 12 * fc.dense_num_params(SMALL_CFG)
        assert d['total'] == sum(v for k, v in d.items() if k != 'total')


# ---------------------------------------------------------------------------
# 4. Subspace analytic (SPEC coverage item 4; exactness per items 1-3 conventions)
# ---------------------------------------------------------------------------

SUBSPACE_INIT_KEYS = {'probe_forward', 'probe_backward', 'probe_optimizer',
                      'probe_eval', 'subspace_build', 'subspace_svd',
                      'factor_propagation', 'total'}


class TestSubspaceAnalytic:
    def test_subspace_width(self):
        assert fc.subspace_width(fc.SubspaceConfig(rank=10)) == 20
        assert fc.subspace_width(SMALL_SCFG) == 4

    def test_num_params_canonical_regression(self):
        # Verified against mlp/logs: 3*20^2 + 784*10 + 20*(784+784) = 40,400.
        scfg = fc.SubspaceConfig(rank=10, train_factors=True)
        assert fc.subspace_num_params(CANON_CFG, scfg) == 40_400

    def test_num_params_frozen_canonical(self):
        scfg = fc.SubspaceConfig(rank=10, train_factors=False)
        assert fc.subspace_num_params(CANON_CFG, scfg) == 9_040  # 1200 + 7840

    def test_num_params_small(self):
        assert fc.subspace_num_params(SMALL_CFG, SMALL_SCFG) == 240
        assert fc.subspace_num_params(SMALL_CFG, SMALL_SCFG_FROZEN) == 112

    def test_fwd_hand_and_closed_form(self):
        assert fc.subspace_fwd_flops(SMALL_CFG, SMALL_SCFG, S_B) == SUB_FWD_SMALL
        # Closed form 2B[d*R + (L-1)*R^2 + R*d_h + d_h*K] on an asymmetric config.
        cfg = fc.MLPConfig(depth=4, in_dim=6, hidden_dim=10, num_classes=3)
        scfg = fc.SubspaceConfig(rank=2)
        R, B = 4, 5
        assert fc.subspace_fwd_flops(cfg, scfg, B) == \
            2 * B * (6 * R + 3 * R * R + R * 10 + 10 * 3)

    def test_fwd_primitive_built(self):
        cfg = fc.MLPConfig(depth=4, in_dim=6, hidden_dim=10, num_classes=3)
        scfg = fc.SubspaceConfig(rank=2)
        B, R = 5, 4
        expected = (fc.linear_fwd_flops(B, 6, R)          # V
                    + 3 * fc.linear_fwd_flops(B, R, R)    # cores
                    + fc.linear_fwd_flops(B, R, 10)       # U^T
                    + fc.linear_fwd_flops(B, 10, 3))      # W_L
        assert fc.subspace_fwd_flops(cfg, scfg, B) == expected

    def test_bwd_hand_small(self):
        assert fc.subspace_bwd_flops(SMALL_CFG, SMALL_SCFG, S_B) == SUB_BWD_TRAIN_SMALL
        assert fc.subspace_bwd_flops(SMALL_CFG, SMALL_SCFG_FROZEN, S_B) == SUB_BWD_FROZEN_SMALL

    def test_bwd_primitive_built_train_factors(self):
        # V: dW only (first layer); cores: dx + dW; U^T: dx + dW; W_L: dx + dW.
        cfg = fc.MLPConfig(depth=4, in_dim=6, hidden_dim=10, num_classes=3)
        scfg = fc.SubspaceConfig(rank=2, train_factors=True)
        B, R = 5, 4
        expected = (fc.linear_bwd_flops(B, 6, R, need_dx=False, need_dw=True)
                    + 3 * fc.linear_bwd_flops(B, R, R)
                    + fc.linear_bwd_flops(B, R, 10)
                    + fc.linear_bwd_flops(B, 10, 3))
        assert fc.subspace_bwd_flops(cfg, scfg, B) == expected

    def test_bwd_primitive_built_frozen_factors(self):
        # Frozen V/U (validated vs the autograd oracle): V contributes nothing; the
        # FIRST core's dx GEMM is pruned (its input comes from frozen V fed by data);
        # U^T keeps dx only; W_L keeps dx + dW.
        cfg = fc.MLPConfig(depth=4, in_dim=6, hidden_dim=10, num_classes=3)
        scfg = fc.SubspaceConfig(rank=2, train_factors=False)
        B, R = 5, 4
        expected = (fc.linear_bwd_flops(B, R, R, need_dx=False, need_dw=True)
                    + 2 * fc.linear_bwd_flops(B, R, R)
                    + fc.linear_bwd_flops(B, R, 10, need_dx=True, need_dw=False)
                    + fc.linear_bwd_flops(B, 10, 3))
        assert fc.subspace_bwd_flops(cfg, scfg, B) == expected

    def test_iter_dict(self):
        d = fc.subspace_iter_flops(SMALL_CFG, SMALL_SCFG, S_B)
        assert set(d) == {'forward', 'backward', 'optimizer', 'total'}
        assert d['forward'] == SUB_FWD_SMALL
        assert d['backward'] == SUB_BWD_TRAIN_SMALL
        assert d['optimizer'] == 2 * 240
        assert d['total'] == d['forward'] + d['backward'] + d['optimizer']

    def test_init_breakdown_structure(self):
        d = fc.subspace_init_flops(CANON_CFG, fc.SubspaceConfig(rank=10), CANON_N)
        assert set(d) == SUBSPACE_INIT_KEYS
        assert d['probe_eval'] == 0  # count_init_eval defaults to False
        for k, v in d.items():
            if k not in ('probe_eval', 'total'):
                assert v > 0, k
        assert d['total'] == sum(v for k, v in d.items() if k != 'total')

    def test_init_count_init_eval(self):
        scfg = fc.SubspaceConfig(rank=10)
        base = fc.subspace_init_flops(CANON_CFG, scfg, CANON_N)
        with_eval = fc.subspace_init_flops(CANON_CFG, scfg, CANON_N, count_init_eval=True)
        assert with_eval['probe_eval'] == fc.dense_fwd_flops(CANON_CFG, CANON_N)
        assert with_eval['probe_eval'] == with_eval['probe_forward']
        assert with_eval['total'] == base['total'] + with_eval['probe_eval']

    def test_init_type1_primitive_built_exact(self):
        cfg = fc.MLPConfig(depth=4, in_dim=6, hidden_dim=10, num_classes=3)
        scfg = fc.SubspaceConfig(rank=2)
        N, R = 25, 4
        expected = {
            'probe_forward': (fc.linear_fwd_flops(N, 6, 10)
                              + 2 * fc.linear_fwd_flops(N, 10, 10)
                              + fc.linear_fwd_flops(N, 10, 3)),
            'probe_backward': (fc.linear_bwd_flops(N, 6, 10, need_dx=False)
                               + 2 * fc.linear_bwd_flops(N, 10, 10)
                               + fc.linear_bwd_flops(N, 10, 3)),
            'probe_optimizer': 2 * (60 + 100 + 100 + 30),
            'probe_eval': 0,
            'subspace_build': fc.matmul_flops(6, 10, 6),          # G1^T @ W1
            'subspace_svd': fc.svd_flops(10 + 6, 6),              # vstack (d_h+d) x d
            'factor_propagation': (fc.matmul_flops(10, 6, R)
                                   + 2 * fc.matmul_flops(10, 10, R)),
        }
        expected['total'] = sum(expected.values())
        assert fc.subspace_init_flops(cfg, scfg, N) == expected

    def test_per_iter_below_dense_canonical(self):
        dense_total = fc.dense_iter_flops(CANON_CFG, CANON_B)['total']
        for tf in (True, False):
            scfg = fc.SubspaceConfig(rank=10, train_factors=tf)
            assert fc.subspace_iter_flops(CANON_CFG, scfg, CANON_B)['total'] < dense_total


# ---------------------------------------------------------------------------
# 3. Oracle: FlopCounterMode on the real repo models (SPEC coverage item 3)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def gmm_data():
    torch.manual_seed(0)
    np.random.seed(0)
    dataset, _, _ = generate_gmm_data(d=S_D, K=S_K, N_k=S_N // S_K)
    x = torch.tensor(dataset.samples[:S_B], dtype=torch.float32)
    assert x.requires_grad is False
    return dataset, x


@pytest.fixture(scope='module')
def dense_model(gmm_data):
    torch.manual_seed(1)
    return MLP(num_layers=S_L, in_dim=S_D, hidden_dim=S_DH, num_classes=S_K,
               activation='GELU', init_method='orthogonal', init_scale=0.1)


@pytest.fixture(scope='module')
def compressed_model(gmm_data):
    """Real CompressedMLP built through compress_model (subspace_type='gradient');
    the 1-epoch full-batch probe train inside is cheap at this size."""
    dataset, _ = gmm_data
    torch.manual_seed(2)
    np.random.seed(2)
    orig = MLP(num_layers=S_L, in_dim=S_D, hidden_dim=S_DH, num_classes=S_K,
               activation='GELU', init_method='orthogonal', init_scale=0.1)
    probe_set = tc.ProbeDataset(dataset.samples, dataset.targets)
    comp = compress_model(orig, rank=S_R, num_classes=S_K, init_method='orthogonal',
                          init_scale=0.1, criterion=nn.MSELoss(reduction='sum'),
                          train_set=probe_set, subspace_type='gradient', device='cpu', angle=0.0)
    assert isinstance(comp, CompressedMLP)
    return comp


class TestOracleFlopCounterMode:
    def test_dense_forward(self, dense_model, gmm_data):
        _, x = gmm_data
        analytic = fc.dense_fwd_flops(SMALL_CFG, S_B)
        measured = tc.count_forward(dense_model, x)
        assert analytic == DENSE_FWD_SMALL
        assert measured == analytic

    def test_dense_forward_backward(self, dense_model, gmm_data):
        _, x = gmm_data
        analytic = fc.dense_fwd_flops(SMALL_CFG, S_B) + fc.dense_bwd_flops(SMALL_CFG, S_B)
        measured = tc.count_forward_backward(dense_model, x)
        assert analytic == DENSE_FWD_SMALL + DENSE_BWD_SMALL
        assert measured == analytic

    def test_compressed_trainable_params(self, compressed_model):
        n_trainable = sum(p.numel() for p in compressed_model.parameters()
                          if p.requires_grad)
        assert n_trainable == fc.subspace_num_params(SMALL_CFG, SMALL_SCFG) == 240

    def test_compressed_forward(self, compressed_model, gmm_data):
        _, x = gmm_data
        analytic = fc.subspace_fwd_flops(SMALL_CFG, SMALL_SCFG, S_B)
        measured = tc.count_forward(compressed_model, x)
        assert analytic == SUB_FWD_SMALL
        assert measured == analytic

    def test_compressed_forward_backward_train_factors(self, compressed_model, gmm_data):
        _, x = gmm_data
        tc.set_factor_requires_grad(compressed_model, True)  # order-independence
        analytic = (fc.subspace_fwd_flops(SMALL_CFG, SMALL_SCFG, S_B)
                    + fc.subspace_bwd_flops(SMALL_CFG, SMALL_SCFG, S_B))
        measured = tc.count_forward_backward(compressed_model, x)
        assert analytic == SUB_FWD_SMALL + SUB_BWD_TRAIN_SMALL
        assert measured == analytic

    def test_compressed_forward_backward_frozen_factors(self, compressed_model, gmm_data):
        # SPEC item 3, frozen variant: requires_grad=False on the two subspace factor
        # Linear layers (identified via the model's subspace_params list).
        _, x = gmm_data
        factors = [entry for entry in compressed_model.subspace_params
                   if isinstance(entry, nn.Linear)]
        assert len(factors) == 2  # V (d -> R) and U^T (R -> d_h)
        n = tc.set_factor_requires_grad(compressed_model, False)
        assert n == 2
        try:
            measured = tc.count_forward_backward(compressed_model, x)
        finally:
            tc.set_factor_requires_grad(compressed_model, True)
        analytic = (fc.subspace_fwd_flops(SMALL_CFG, SMALL_SCFG_FROZEN, S_B)
                    + fc.subspace_bwd_flops(SMALL_CFG, SMALL_SCFG_FROZEN, S_B))
        assert analytic == SUB_FWD_SMALL + SUB_BWD_FROZEN_SMALL
        assert measured == analytic


# ---------------------------------------------------------------------------
# 5. LTH (SPEC coverage item 5)
# ---------------------------------------------------------------------------

class TestLTH:
    def test_density_formula(self):
        lcfg = fc.LTHConfig(prune_rate=0.2)
        assert fc.lth_density(lcfg, 0) == 1.0
        assert fc.lth_density(lcfg, 1) == 0.8
        for k in range(2, 6):
            assert fc.lth_density(lcfg, k) == (1 - 0.2) ** k

    def test_rounds_for_density_inverse(self):
        # Smallest n with (1-p)^n <= target; exact inverse on exact powers.
        for k in range(1, 13):
            assert fc.lth_rounds_for_density(0.2, 0.8 ** k) == k
        assert fc.lth_rounds_for_density(0.5, 0.3) == 2   # 0.5 > 0.3, 0.25 <= 0.3
        assert fc.lth_rounds_for_density(0.5, 0.5) == 1
        assert fc.lth_rounds_for_density(0.2, 1.0) == 1   # n >= 1

    def test_theoretical_iter_exact(self):
        lcfg = fc.LTHConfig(prune_rate=0.2, theoretical_sparsity=True)
        p_dense = fc.dense_num_params(SMALL_CFG)
        for density in (1.0, 0.5, 0.37):
            d = fc.lth_iter_flops(SMALL_CFG, lcfg, S_B, density)
            assert set(d) == {'forward', 'backward', 'optimizer', 'mask_apply', 'total'}
            assert d['forward'] == int(round(density * fc.dense_fwd_flops(SMALL_CFG, S_B)))
            assert d['backward'] == int(round(density * fc.dense_bwd_flops(SMALL_CFG, S_B)))
            assert d['optimizer'] == 2 * math.ceil(density * p_dense)
            assert d['mask_apply'] == 0
            assert d['total'] == sum(v for k, v in d.items() if k != 'total')

    def test_theoretical_iter_full_density_equals_dense(self):
        lcfg = fc.LTHConfig(theoretical_sparsity=True)
        d = fc.lth_iter_flops(SMALL_CFG, lcfg, S_B, 1.0)
        assert d['forward'] == fc.dense_fwd_flops(SMALL_CFG, S_B)
        assert d['backward'] == fc.dense_bwd_flops(SMALL_CFG, S_B)
        assert d['optimizer'] == 2 * fc.dense_num_params(SMALL_CFG)

    def test_theoretical_iter_scales_with_density(self):
        lcfg = fc.LTHConfig(theoretical_sparsity=True)
        totals = [fc.lth_iter_flops(SMALL_CFG, lcfg, S_B, rho)['total']
                  for rho in (0.25, 0.5, 1.0)]
        assert totals[0] < totals[1] < totals[2]

    def test_masked_iter_exact_and_geq_dense(self):
        lcfg = fc.LTHConfig(theoretical_sparsity=False)
        p_dense = fc.dense_num_params(SMALL_CFG)
        d = fc.lth_iter_flops(SMALL_CFG, lcfg, S_B, 0.1)
        assert d['forward'] == fc.dense_fwd_flops(SMALL_CFG, S_B)
        assert d['backward'] == fc.dense_bwd_flops(SMALL_CFG, S_B)
        assert d['optimizer'] == 2 * p_dense
        assert d['mask_apply'] == p_dense
        assert d['total'] == sum(v for k, v in d.items() if k != 'total')
        assert d['total'] >= (fc.dense_fwd_flops(SMALL_CFG, S_B)
                              + fc.dense_bwd_flops(SMALL_CFG, S_B))
        assert d['total'] == fc.dense_iter_flops(SMALL_CFG, S_B)['total'] + p_dense

    def test_search_hand_built_exact(self):
        # Tiny config, p=0.5, n=2 rounds, T=3 iterations, B=4: rebuild the SPEC sum.
        cfg = fc.MLPConfig(depth=3, in_dim=6, hidden_dim=5, num_classes=2)
        lcfg = fc.LTHConfig(prune_rate=0.5, n_rounds=2, iters_per_round=3,
                            theoretical_sparsity=True)
        B = 4
        training = 0
        for i in range(2):
            density_i = 0.5 ** i
            training += 3 * fc.lth_iter_flops(cfg, lcfg, B, density_i)['total']
        expected = {'training': training, 'total': training}
        assert fc.lth_search_flops(cfg, lcfg, B) == expected

    def test_search_one_shot_cheaper_than_iterative(self):
        one = fc.LTHConfig(prune_rate=0.2, n_rounds=1, iters_per_round=100)
        three = fc.LTHConfig(prune_rate=0.2, n_rounds=3, iters_per_round=100)
        assert fc.lth_search_flops(SMALL_CFG, one, S_B)['total'] < \
            fc.lth_search_flops(SMALL_CFG, three, S_B)['total']

    def test_final_iter_equals_iter_at_final_density(self):
        lcfg = fc.LTHConfig(prune_rate=0.2, n_rounds=5, iters_per_round=7)
        assert fc.lth_final_iter_flops(SMALL_CFG, lcfg, S_B) == \
            fc.lth_iter_flops(SMALL_CFG, lcfg, S_B, fc.lth_density(lcfg, 5))


# ---------------------------------------------------------------------------
# 6. DLRT (SPEC coverage item 6)
# ---------------------------------------------------------------------------

DLRT_ITER_KEYS = {'kl_fwd', 'kl_bwd', 's_fwd', 's_bwd', 'optimizer', 'total'}


def _dlrt_expected_from_spec(cfg, r, adaptive, B, opt_fpp=2):
    """Independent transcription of the SPEC dlrt_iter_flops formulas."""
    R2 = 2 * r if adaptive else r
    dims = fc.dense_layer_dims(cfg)[:-1]
    d_h, K_cls = cfg.hidden_dim, cfg.num_classes
    kl_fwd = kl_bwd = s_fwd = s_bwd = 0
    for i, (out, in_) in enumerate(dims):
        first = (i == 0)
        kl_fwd += 2 * B * r * in_ + 2 * B * r * out
        if not first:
            kl_bwd += 2 * B * r * out + 2 * B * r * in_
        kl_bwd += 2 * B * out * r                      # dK = dy zhat^T
        kl_bwd += 2 * B * out * r + 2 * B * in_ * r    # dL via w = U^T dy, z w^T
        s_fwd += 2 * B * R2 * in_ + 2 * B * R2 ** 2 + 2 * B * R2 * out
        s_bwd += 2 * B * out * R2 + 2 * B * R2 ** 2
        if not first:
            s_bwd += 2 * B * R2 ** 2 + 2 * B * in_ * R2
    kl_fwd += 2 * B * d_h * K_cls
    kl_bwd += 2 * B * d_h * K_cls
    s_fwd += 2 * B * d_h * K_cls
    s_bwd += 4 * B * d_h * K_cls
    optimizer = opt_fpp * (sum(o * r + i_ * r + R2 ** 2 for (o, i_) in dims)
                           + d_h * K_cls)
    d = {'kl_fwd': kl_fwd, 'kl_bwd': kl_bwd, 's_fwd': s_fwd, 's_bwd': s_bwd,
         'optimizer': optimizer}
    d['total'] = sum(d.values())
    return d


class TestDLRT:
    def test_num_params_hand(self):
        # L=3, d=7, d_h=5, K=2, r=3: (3*12+9) + (3*10+9) + 10 = 94.
        cfg = fc.MLPConfig(depth=3, in_dim=7, hidden_dim=5, num_classes=2)
        assert fc.dlrt_num_params(cfg, fc.DLRTConfig(rank=3)) == 94

    def test_iter_breakdown_keys_and_total(self):
        d = fc.dlrt_iter_flops(CANON_CFG, fc.DLRTConfig(rank=20), CANON_B)
        assert set(d) == DLRT_ITER_KEYS
        assert d['total'] == sum(v for k, v in d.items() if k != 'total')

    def test_iter_hand_built_adaptive_single_low_rank_layer(self):
        # depth=2, d=6, d_h=5, K=3, r=2 (R2=4), B=7 -- every component by hand.
        cfg = fc.MLPConfig(depth=2, in_dim=6, hidden_dim=5, num_classes=3)
        got = fc.dlrt_iter_flops(cfg, fc.DLRTConfig(rank=2, adaptive=True), 7)
        assert got == {'kl_fwd': 518, 'kl_bwd': 658, 's_fwd': 1050, 's_bwd': 924,
                       'optimizer': 106, 'total': 3256}

    def test_iter_hand_built_fixed_rank_single_low_rank_layer(self):
        cfg = fc.MLPConfig(depth=2, in_dim=6, hidden_dim=5, num_classes=3)
        got = fc.dlrt_iter_flops(cfg, fc.DLRTConfig(rank=2, adaptive=False), 7)
        assert got == {'kl_fwd': 518, 'kl_bwd': 658, 's_fwd': 574, 's_bwd': 616,
                       'optimizer': 82, 'total': 2448}

    def test_iter_spec_formula_depth3_both_variants(self):
        # Exercises the non-first-layer dz paths for adaptive and fixed rank.
        cfg = fc.MLPConfig(depth=3, in_dim=7, hidden_dim=5, num_classes=3)
        for adaptive in (True, False):
            dcfg = fc.DLRTConfig(rank=2, adaptive=adaptive)
            assert fc.dlrt_iter_flops(cfg, dcfg, 4) == \
                _dlrt_expected_from_spec(cfg, 2, adaptive, 4)

    def test_adaptive_iter_greater_than_fixed_canonical(self):
        adaptive = fc.dlrt_iter_flops(CANON_CFG, fc.DLRTConfig(rank=20, adaptive=True),
                                      CANON_B)
        fixed = fc.dlrt_iter_flops(CANON_CFG, fc.DLRTConfig(rank=20, adaptive=False),
                                   CANON_B)
        assert adaptive['total'] > fixed['total']

    def test_per_iter_below_dense_canonical(self):
        dense_total = fc.dense_iter_flops(CANON_CFG, CANON_B)['total']
        for dcfg in (fc.DLRTConfig(rank=20, adaptive=True),
                     fc.DLRTConfig(rank=20, adaptive=False)):
            assert fc.dlrt_iter_flops(CANON_CFG, dcfg, CANON_B)['total'] < dense_total

    def test_init_hand_built_exact(self):
        # depth=2, d=6, d_h=5, K=3: low-rank layer (5,6), dense last (3,5).
        cfg = fc.MLPConfig(depth=2, in_dim=6, hidden_dim=5, num_classes=3)
        assert fc.dlrt_init_flops(cfg, fc.DLRTConfig(rank=2)) == \
            {'warm_start': 0, 'total': 0}
        # Warm start: svd(6,5) = 14*6*25 + 8*125 = 3100.
        got = fc.dlrt_init_flops(cfg, fc.DLRTConfig(rank=2, warm_start_svd=True))
        assert got == {'warm_start': 3100, 'total': 3100}

    def test_init_warm_start_greater(self):
        base = fc.dlrt_init_flops(CANON_CFG, fc.DLRTConfig(rank=20))
        warm = fc.dlrt_init_flops(CANON_CFG, fc.DLRTConfig(rank=20, warm_start_svd=True))
        assert warm['total'] > base['total']
        assert base['warm_start'] == 0
        assert base['total'] == base['warm_start']
        assert warm['total'] == warm['warm_start']

    def test_oracle_kstep_forward(self, gmm_data):
        # SPEC item 6 oracle / torch_check item 4: per low-rank layer a frozen
        # Linear(in->r) followed by a trainable Linear(r->out), plus the trainable
        # dense last Linear. Forward FLOPs == primitive-built sum == 'kl_fwd'.
        _, x = gmm_data
        torch.manual_seed(3)
        cfg3 = fc.MLPConfig(depth=3, in_dim=S_D, hidden_dim=S_DH, num_classes=S_K)
        low_rank_dims = fc.dense_layer_dims(cfg3)[:-1]
        net = tc.KStepNet(low_rank_dims, rank=S_R, hidden_dim=S_DH, num_classes=S_K)
        frozen = [p for p in net.parameters() if not p.requires_grad]
        assert len(frozen) == len(low_rank_dims)  # one frozen V per low-rank layer
        primitives = sum(fc.linear_fwd_flops(S_B, in_, S_R)
                         + fc.linear_fwd_flops(S_B, S_R, out)
                         for (out, in_) in low_rank_dims)
        primitives += fc.linear_fwd_flops(S_B, S_DH, S_K)
        assert primitives == KL_FWD_SMALL
        assert primitives == fc.dlrt_iter_flops(cfg3, fc.DLRTConfig(rank=S_R),
                                                S_B)['kl_fwd']
        measured = tc.count_forward(net, x)
        assert measured == primitives


# ---------------------------------------------------------------------------
# 7. compare.py (SPEC coverage item 7)
# ---------------------------------------------------------------------------

SMALL_ARGV = ['--depth', '4', '--in_dim', '16', '--hidden_dim', '16',
              '--num_classes', '4', '--rank', '2', '--num_train_samples', '64',
              '--dense_epochs', '5', '--comp_epochs', '10']


def _parse(argv):
    return compare.build_parser().parse_args(argv)


def _assert_breakdown_ints(d):
    """Every entry a non-negative int (bool excluded); 'total' == sum of the rest."""
    assert 'total' in d
    for k, v in d.items():
        assert type(v) is int, (k, type(v))
        assert v >= 0, k
    assert d['total'] == sum(v for k, v in d.items() if k != 'total')


class TestCompare:
    def test_build_comparison_structure(self):
        result = compare.build_comparison(_parse(SMALL_ARGV))
        assert set(result) == {'config', 'schemes'}
        assert set(result['schemes']) == set(compare.SCHEME_ORDER)
        assert len(result['schemes']) == 6
        for name in compare.SCHEME_ORDER:
            row = result['schemes'][name]
            _assert_breakdown_ints(row['init'])
            _assert_breakdown_ints(row['per_iter'])
            for key in ('trainable_params', 'epochs', 'init_total',
                        'per_iter_total', 'run_total'):
                assert type(row[key]) is int and row[key] >= 0, (name, key)
            assert row['init_total'] == row['init']['total']
            assert row['per_iter_total'] == row['per_iter']['total']
            assert row['run_total'] == \
                row['init_total'] + row['epochs'] * row['per_iter_total']
            for key in ('per_iter_vs_dense', 'run_vs_dense'):
                assert math.isfinite(row[key]), (name, key)

    def test_build_comparison_canonical_numbers(self):
        result = compare.build_comparison(_parse([]))  # canonical defaults
        schemes = result['schemes']
        assert schemes['Dense']['trainable_params'] == 1_851_808
        assert schemes['Subspace (trained factors)']['trainable_params'] == 40_400
        assert schemes['Subspace (frozen factors)']['trainable_params'] == 9_040
        # LTH auto rounds prune to the subspace parameter budget; LTH row param count
        # = ceil(final_density * P_dense) (implemented behavior).
        n = fc.lth_rounds_for_density(0.2, 40_400 / 1_851_808)
        assert result['config']['lth_rounds'] == n
        lcfg = fc.LTHConfig(prune_rate=0.2, n_rounds=n)
        assert schemes['LTH']['trainable_params'] == \
            math.ceil(fc.lth_density(lcfg, n) * 1_851_808)
        assert result['config']['dlrt_rank'] == 20

    def test_main_writes_results_json(self, tmp_path):
        argv = SMALL_ARGV + ['--out_dir', str(tmp_path)]
        assert compare.main(argv) == 0
        json_path = tmp_path / 'results.json'
        assert json_path.is_file()
        loaded = json.loads(json_path.read_text())
        expected = compare.build_comparison(_parse(argv))
        assert set(loaded['schemes']) == set(compare.SCHEME_ORDER)
        assert loaded['config'] == expected['config']
        for name in compare.SCHEME_ORDER:
            got, want = loaded['schemes'][name], expected['schemes'][name]
            for key in ('trainable_params', 'epochs', 'init_total',
                        'per_iter_total', 'run_total'):
                assert got[key] == want[key], (name, key)
            assert got['init'] == want['init']
            assert got['per_iter'] == want['per_iter']


# ---------------------------------------------------------------------------
# 8. Break-even sanity, canonical config (SPEC coverage item 8)
# ---------------------------------------------------------------------------

class TestBreakEvenCanonical:
    def test_subspace_init_below_5pct_of_lth_search(self):
        scfg = fc.SubspaceConfig(rank=10, train_factors=True)
        target = fc.subspace_num_params(CANON_CFG, scfg) / fc.dense_num_params(CANON_CFG)
        n = fc.lth_rounds_for_density(0.2, target)
        lcfg = fc.LTHConfig(prune_rate=0.2, n_rounds=n, iters_per_round=500,
                            theoretical_sparsity=True)
        search_total = fc.lth_search_flops(CANON_CFG, lcfg, CANON_B)['total']
        init_total = fc.subspace_init_flops(CANON_CFG, scfg, CANON_N)['total']
        assert init_total < 0.05 * search_total

    def test_subspace_per_iter_below_10pct_of_dense(self):
        dense_total = fc.dense_iter_flops(CANON_CFG, CANON_B)['total']
        for tf in (True, False):
            scfg = fc.SubspaceConfig(rank=10, train_factors=tf)
            total = fc.subspace_iter_flops(CANON_CFG, scfg, CANON_B)['total']
            assert total < 0.10 * dense_total
