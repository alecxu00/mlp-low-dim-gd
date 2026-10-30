# mlp-low-dim-gd

Code for [Emergent Low-Rank Training Dynamics in MLPs with Smooth Activations](https://arxiv.org/abs/2602.06208)

| Directory | Contents |
|---|---|
| [`mlp/`](mlp/README.md) | MLPs on Fashion-MNIST, plus the theory notebooks |
| [`cnn/`](cnn/README.md) | VGG-16 classifier head on CIFAR-10 |
| [`flop_comparison/`](flop_comparison/README.md) | FLOP comparison against dense training, LTH and DLRT |

## Setup

Python 3.12:

```bash
pip install -r requirements.txt
```

## Usage

Scripts in `*/scripts/` run under `bash` or `sbatch`, from inside that directory. Results and logs go to `*/results/` and `*/logs/`, which are gitignored. Tests: `python3 -m pytest flop_comparison/tests`.