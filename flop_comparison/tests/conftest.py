"""sys.path bootstrap for the flop_comparison test suite.

Inserts (derived from __file__, never from the CWD):
  1. the flop_comparison/ directory  -> flop_counter, compare, torch_check
  2. <repo>/mlp                      -> model, data, train_utils, utils (real repo code)

This makes the suite runnable both from the repo root
(`python3 -m pytest flop_comparison/tests -q`) and from flop_comparison/scripts/
(`python3 -m pytest ../tests -q`, as the SLURM script does).
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_FLOP_DIR = os.path.dirname(_TESTS_DIR)                    # flop_comparison/
_REPO_DIR = os.path.dirname(_FLOP_DIR)                     # repo root
_MLP_DIR = os.path.join(_REPO_DIR, 'mlp')

for _p in (_MLP_DIR, _FLOP_DIR):                           # _FLOP_DIR ends up first
    if _p not in sys.path:
        sys.path.insert(0, _p)
