# VGG-16 on CIFAR-10

Compresses the classifier head of an ImageNet-pretrained VGG-16 fine-tuned on CIFAR-10. CIFAR-100 is also supported.

## Run

```bash
cd cnn/scripts
sbatch train_full_cnn.sh   # uncompressed classifier
sbatch train_comp_cnn.sh   # low rank classifier
```

- **What trains:** both scripts train only the classifier (`--freeze_conv`, which also keeps the conv BatchNorm statistics fixed); remove that flag to train the whole network.
- **Requirements:** needs a GPU. All flags: `python3 ../train_cnn.py --help`.
- **Data download:** download CIFAR once before launching several jobs. The official host is slow, and parallel jobs would race to download it:
  ```bash
  python3 -c "from torchvision.datasets import CIFAR10; [CIFAR10('<data_dir>/cifar10', train=t, download=True) for t in (True, False)]"
  ```

- **Data:** first `K` classes, `N_k` images per class without replacement, shuffled and resized to 224×224.
- **Model:** `vgg16_bn` (ImageNet conv weights with `--pt_conv`), average pool resized to 3×3, then a classifier `4608 → 4096 → 4096 → K` using `--activation`.
- **Training:** SGD (momentum 0.9, weight decay 5e-4), cosine learning-rate schedule down to `lr/100`; batch 128 in the scripts.
- **Runtime** (5 trials × 250 epochs): about 4.5 days for the full model, 2.6 days classifier-only.

## Compression (`--compress`)

The classifier becomes `V (4608→2r) → core → core → U (2r→4096) → 4096→K`.
- **`gradient`:** a one-minibatch probe step on a copy of the model gives `G1(0)`. `V` is built from it, and `U = W2(0) W1(0) V / ε²`.
- **`random`** (the default): random bases.

All of the classifier's parameters train.

## Outputs

`results/<data>_K_<K>_N_<N>/<act>_vgg16[_frozen_conv]_<init>_init_scale_<s>_eta_<lr>/{orig_model|compressed_model_rank_<r>_subspace_<type>}/trial_<i>_{losses,accs}.npz`

- `*_all_trials.npz` files are also written at the end.
- `plot_losses_cnn.ipynb` reads these paths.