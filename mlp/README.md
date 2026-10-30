# MLP on Fashion-MNIST

Trains deep MLPs on Fashion-MNIST, either full or compressed into a low-dimensional subspace fixed at initialization. Also holds the theory notebooks, which use synthetic data.

## Run

```bash
cd mlp/scripts
bash train_full_mlp.sh   # full MLP
bash train_comp_mlp.sh   # low-rank MLP
```

Run from inside `scripts/`, and set hyperparameters with the shell variables at the top of each script (`sbatch` works too). All flags: `python3 ../train_mlp.py --help`. Fashion-MNIST downloads automatically; pass `--data_dir` to change where it's stored.

- **Data:** first `K` classes, `N_k` images per class sampled without replacement and shuffled; flattened to 784 values, one-hot labels.
- **Model:** `--depth` bias-free linear layers, `784 → d_h → … → K`, orthogonal init at `--init_scale`.
- **Training:** full-batch SGD (or `--batch_size`), cosine learning-rate schedule down to `lr/100`, sum-reduced loss (divide by `N` to compare runs).

## Compression (`--compress`)

A one-step probe gives `W(0)` and the first-layer gradient `G1(0)`.
- **`--subspace_type gradient`:** builds `2r`-dimensional bases from them, with `r = --compress_rank` (default `K`). `--subspace_angle` rotates them away from that subspace.
- **`random`** (the default): uses random bases.

The compressed network is `V_1ᵀ (784→2r) → L−1 cores (2r×2r) → U_{L−1} (2r→d_h) → d_h→K`, and all of its parameters train. That's 40,400 parameters, against 1,851,808 for the full canonical MLP. The last layer starts at the full model's `W_L(0)`.

## Outputs

`results/fashion_mnist_d_<d>_K_<K>_N_<N>/<act>_network_L_<L>_<init>_init_scale_<s>_eta_<lr>/{orig_model|compressed_model_subspace_<type>}_{losses,accs}.npz`

- Each array has shape `(num_trials, epochs)`.
- `plot_losses.ipynb` reads these paths.
- Hidden width, batch size, rank and angle aren't part of the path.

## Notebooks

Run with `mlp/` as the working directory.

- `verify_theory`: singular-subspace drift of the weights during training
- `assumption`: checks the theory's assumptions
- `beyond_theory`: regimes outside the assumptions
- `theory_W2_G1_subspaces`: `G1` subspaces with a frozen vs. trained last layer
- `main_fig`: main paper figure
- `plot_losses`: loss and accuracy curves