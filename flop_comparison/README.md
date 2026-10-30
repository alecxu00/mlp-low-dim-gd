# flop_comparison/

Analytic FLOP comparison of the one-time **initialization cost** and the **per-iteration
training cost** of three low-dimensional training schemes against the dense MLP baseline
trained in `mlp/train_mlp.py`.

## Schemes

- **Dense MLP** — the fully parameterized `mlp/model.py:MLP` (bias=False).
- **Subspace-compressed (this repo)** — `compress_model` + `CompressedMLP`.
- **LTH** (Frankle & Carbin, ICLR 2019, Sec. 1/2)
- **DLRT** (Schotthoefer et al., NeurIPS 2022, Algorithm 1 + Sec. 4.3)

## Conventions (all counts follow these; see `flop_counter.py`)

- 1 MAC = 2 FLOPs; every (m x k)(k x n) matmul costs `2*m*k*n`.
- **Matmul-only**: elementwise activations, losses and bias terms are excluded (repo uses
  bias=False) — this matches `torch.utils.flop_counter.FlopCounterMode`.
- Linear backward: dx GEMM `2*B*in*out`, **skipped whenever the layer input requires no
  gradient**.
- Optimizer: 2 FLOPs per trainable param per iteration (plain SGD)
- Economy SVD with both singular-vector matrices (Golub–Van Loan): `14*m*n^2 + 8*n^3`.
- Not counted: building initial weights (orthogonal-init QR), the LTH magnitude-prune
  sort, and DLRT's per-step overhead (K/L forming, QRs, basis projections, rank-truncation SVD).
- Every breakdown dict carries a `'total'` key equal to the sum of its other entries.

## Results

| Scheme | Trainable params | Init FLOPs | Per-iter FLOPs |
|---|---:|---:|---:|
| Dense                                  | 1,237,152 | N/A | 1.59e+09 |
| Subspace                               | 40,000    | 1.99e+10 | 5.35e+07 |
| LTH (one-shot, p=0.9677)               | 40,001    | 1.59e+09·T | 5.13e+07 |
| DLRT (adaptive, r=10, warm start)      | 39,400    | 2.12e+10 | 1.26e+08 |
| DLRT (fixed-rank, r=10, warm start)    | 39,400    | 2.12e+10 | 9.67e+07 |

Config: `L=3, d=d_h=784, K=10, N=B=256`. Regenerate with the command below. It prints the LTH init at T = `--lth_iters_per_round`, which defaults to 1.

```bash
python3 compare.py --depth 3 --rank 10 --num_train_samples 256 --batch_size 256 \
  --lth_prune_rate 0.967667 --lth_rounds 1 --dlrt_rank 10 --dlrt_warm_start
```

## How to run

```bash
cd flop_comparison

# Analytic table + results.json (+ PDF figure); all flags optional, defaults = canonical.
python3 compare.py --save_fig --out_dir ./results
python3 compare.py --help     # depth/in_dim/hidden_dim/num_classes/rank,
                              # num_train_samples/batch_size/opt, dense_epochs/comp_epochs,
                              # lth_prune_rate/lth_rounds/lth_iters_per_round,
                              # dlrt_rank/dlrt_warm_start

# Empirical validation vs FlopCounterMode (exit 0 iff every comparison matches exactly).
python3 torch_check.py --small --device auto   # quick CPU config
python3 torch_check.py --device auto

# Tests
python3 -m pytest tests -q                     # or from the repo root: pytest flop_comparison/tests -q

# Full pipeline on the cluster (or plain bash), from inside scripts/:
cd scripts && sbatch run_flop_comparison.sh    # or: bash run_flop_comparison.sh
```

## Caveats

- **LTH FLOPs are theoretical**: unstructured sparsity is counted as `density * dense`
  (idealized sparse kernels). Real dense hardware executes masked-dense — use
  `LTHConfig(theoretical_sparsity=False)` for that cost model (full dense fwd/bwd + one
  mask multiply per weight).
- **DLRT constants** follow the paper's Algorithm 1 with the K- and L-step sharing one
  factored forward pass; bias updates are omitted (bias=False everywhere here).
- **The repo's U/V factors train** (the logged 40,400-param runs); `compare.py` also
  reports a frozen-factor variant as a separate row, not shown in the table above. In the frozen variant the
  first core's dx GEMM is skipped (matching PyTorch autograd exactly — verified against
  `FlopCounterMode` in `torch_check.py`).
- **Init costs are one-time** and amortize over training: e.g. the subspace init
  (~12.6 dense iterations, mostly the one SVD) is repaid after ~13 iterations of the
  ~30x cheaper compressed updates.
- Elementwise ops (activations, losses, masks in theoretical mode) are excluded
  everywhere, so absolute numbers slightly undercount real work; ratios are unaffected.
