"""
    CLI comparing init and per-iteration FLOPs of the dense MLP, the repo's
    subspace-compressed scheme, LTH iterative magnitude pruning, and DLRT.
"""

import argparse
import json
import math
import os

import flop_counter as fc

SCHEME_ORDER = [
    'Dense',
    'Subspace (trained factors)',
    'Subspace (frozen factors)',
    'LTH',
    'DLRT (adaptive)',
    'DLRT (fixed-rank)',
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Analytic FLOP comparison of low-dimensional training schemes '
                    '(defaults = the canonical mlp/ experiment config).')
    p.add_argument('--depth', type=int, default=4, help='L, number of Linear layers')
    p.add_argument('--in_dim', type=int, default=784, help='d, input dimension')
    p.add_argument('--hidden_dim', type=int, default=784, help='d_h, hidden dimension')
    p.add_argument('--num_classes', type=int, default=10, help='K, output dimension')
    p.add_argument('--rank', type=int, default=10,
                   help='r for the subspace scheme (width R = 2r)')
    p.add_argument('--num_train_samples', type=int, default=256, help='N')
    p.add_argument('--batch_size', type=int, default=None,
                   help='B per iteration (default: full batch = N)')
    p.add_argument('--opt', choices=['sgd', 'adam'], default='sgd',
                   help='optimizer cost model: sgd -> 2, adam -> 12 FLOPs/param')
    p.add_argument('--dense_epochs', type=int, default=1,
                   help='epochs for dense / LTH ticket / DLRT full runs')
    p.add_argument('--comp_epochs', type=int, default=1,
                   help='epochs for the subspace-compressed full run')
    p.add_argument('--lth_prune_rate', type=float, default=0.2,
                   help='p, fraction pruned per LTH round')
    p.add_argument('--lth_rounds', type=int, default=None,
                   help='LTH rounds n (default: auto, prune to the subspace '
                        'scheme parameter budget)')
    p.add_argument('--lth_iters_per_round', type=int, default=None,
                   help='T iterations per LTH search round (default: dense_epochs)')
    p.add_argument('--dlrt_rank', type=int, default=None,
                   help='DLRT steady-state rank (default: 2*rank)')
    p.add_argument('--dlrt_warm_start', action='store_true',
                   help='DLRT init from the dense starting weights (paper Sec. 5.2): '
                        'adds one economy SVD per low-rank layer, '
                        'instead of the default random-factor init')
    p.add_argument('--out_dir', default='./results')
    p.add_argument('--save_fig', action='store_true',
                   help='also write a log-scale grouped bar chart PDF')
    return p


def _row(trainable_params: int, epochs: int, init: dict, per_iter: dict) -> dict:
    return {'trainable_params': trainable_params,
            'epochs': epochs,
            'init': init,
            'per_iter': per_iter,
            'init_total': init['total'],
            'per_iter_total': per_iter['total'],
            'run_total': init['total'] + epochs * per_iter['total']}


def build_comparison(args) -> dict:
    """Assemble the six scheme rows (SPEC table) and the resolved config."""
    cfg = fc.MLPConfig(depth=args.depth, in_dim=args.in_dim,
                       hidden_dim=args.hidden_dim, num_classes=args.num_classes)
    batch = args.batch_size if args.batch_size is not None else args.num_train_samples
    opt_fpp = {'sgd': 2, 'adam': 12}[args.opt]

    scfg_impl = fc.SubspaceConfig(rank=args.rank, train_factors=True)
    scfg_frozen = fc.SubspaceConfig(rank=args.rank, train_factors=False)

    # LTH: default rounds prune down to the subspace scheme's parameter budget.
    target_density = fc.subspace_num_params(cfg, scfg_impl) / fc.dense_num_params(cfg)
    n_rounds = (args.lth_rounds if args.lth_rounds is not None
                else fc.lth_rounds_for_density(args.lth_prune_rate, target_density))
    iters_per_round = (args.lth_iters_per_round if args.lth_iters_per_round is not None
                       else args.dense_epochs)
    lcfg = fc.LTHConfig(prune_rate=args.lth_prune_rate, n_rounds=n_rounds,
                        iters_per_round=iters_per_round, theoretical_sparsity=True)

    dlrt_rank = args.dlrt_rank if args.dlrt_rank is not None else 2 * args.rank
    warm_start = getattr(args, 'dlrt_warm_start', False)
    dcfg_adaptive = fc.DLRTConfig(rank=dlrt_rank, adaptive=True,
                                  warm_start_svd=warm_start)
    dcfg_fixed = fc.DLRTConfig(rank=dlrt_rank, adaptive=False,
                               warm_start_svd=warm_start)

    schemes = {
        'Dense': _row(
            fc.dense_num_params(cfg), args.dense_epochs, {'total': 0},
            fc.dense_iter_flops(cfg, batch, opt_fpp)),
        'Subspace (trained factors)': _row(
            fc.subspace_num_params(cfg, scfg_impl), args.comp_epochs,
            fc.subspace_init_flops(cfg, scfg_impl, args.num_train_samples, opt_fpp),
            fc.subspace_iter_flops(cfg, scfg_impl, batch, opt_fpp)),
        'Subspace (frozen factors)': _row(
            fc.subspace_num_params(cfg, scfg_frozen), args.comp_epochs,
            fc.subspace_init_flops(cfg, scfg_frozen, args.num_train_samples, opt_fpp),
            fc.subspace_iter_flops(cfg, scfg_frozen, batch, opt_fpp)),
        'LTH': _row(
            math.ceil(fc.lth_density(lcfg, n_rounds) * fc.dense_num_params(cfg)),
            args.dense_epochs,
            fc.lth_search_flops(cfg, lcfg, batch, opt_fpp),
            fc.lth_final_iter_flops(cfg, lcfg, batch, opt_fpp)),
        'DLRT (adaptive)': _row(
            fc.dlrt_num_params(cfg, dcfg_adaptive), args.dense_epochs,
            fc.dlrt_init_flops(cfg, dcfg_adaptive),
            fc.dlrt_iter_flops(cfg, dcfg_adaptive, batch, opt_fpp)),
        'DLRT (fixed-rank)': _row(
            fc.dlrt_num_params(cfg, dcfg_fixed), args.dense_epochs,
            fc.dlrt_init_flops(cfg, dcfg_fixed),
            fc.dlrt_iter_flops(cfg, dcfg_fixed, batch, opt_fpp)),
    }

    dense_row = schemes['Dense']
    for row in schemes.values():
        row['per_iter_vs_dense'] = row['per_iter_total'] / dense_row['per_iter_total']
        row['run_vs_dense'] = row['run_total'] / dense_row['run_total']

    config = {'depth': args.depth, 'in_dim': args.in_dim, 'hidden_dim': args.hidden_dim,
              'num_classes': args.num_classes, 'rank': args.rank,
              'num_train_samples': args.num_train_samples, 'batch_size': batch,
              'opt': args.opt, 'opt_flops_per_param': opt_fpp,
              'dense_epochs': args.dense_epochs, 'comp_epochs': args.comp_epochs,
              'lth_prune_rate': args.lth_prune_rate, 'lth_rounds': n_rounds,
              'lth_iters_per_round': iters_per_round,
              'lth_final_density': fc.lth_density(lcfg, n_rounds),
              'dlrt_rank': dlrt_rank, 'dlrt_warm_start': warm_start}
    return {'config': config, 'schemes': schemes}


# ---------------------------------------------------------------------------
# Formatting / output
# ---------------------------------------------------------------------------

def _human(x: float) -> str:
    for thresh, suffix in [(1e18, 'EF'), (1e15, 'PF'), (1e12, 'TF'),
                           (1e9, 'GF'), (1e6, 'MF'), (1e3, 'kF')]:
        if x >= thresh:
            return f'{x / thresh:.2f} {suffix}'
    return f'{x:.0f} F'


def fmt_flops(x: int) -> str:
    return f'{x:.2e} ({_human(x)})'


def print_table(result: dict) -> None:
    headers = ['Scheme', 'Params', 'Init FLOPs', 'Per-iter FLOPs', 'Full-run FLOPs',
               'Iter/Dense', 'Run/Dense']
    rows = []
    for name in SCHEME_ORDER:
        r = result['schemes'][name]
        init = fmt_flops(r['init_total']) if r['init_total'] else 'N/A'
        rows.append([name, f"{r['trainable_params']:,}",
                     init, fmt_flops(r['per_iter_total']),
                     fmt_flops(r['run_total']),
                     f"{r['per_iter_vs_dense']:.4g}x", f"{r['run_vs_dense']:.4g}x"])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = '  '
    print(sep.join(h.ljust(w) for h, w in zip(headers, widths)))
    print(sep.join('-' * w for w in widths))
    for row in rows:
        print(sep.join(cell.ljust(w) for cell, w in zip(row, widths)))


def print_breakdowns(result: dict) -> None:
    for name in SCHEME_ORDER:
        r = result['schemes'][name]
        print(f'\n{name} (params={r["trainable_params"]:,}, epochs={r["epochs"]})')
        print('  init:')
        for k, v in r['init'].items():
            print(f'    {k:<20} {fmt_flops(v)}')
        print('  per-iteration:')
        for k, v in r['per_iter'].items():
            print(f'    {k:<20} {fmt_flops(v)}')


def save_figure(result: dict, path: str) -> None:
    """Log-scale grouped bar chart (init vs per-iteration FLOPs) as a PDF."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = SCHEME_ORDER
    init_vals = [result['schemes'][n]['init_total'] for n in names]
    iter_vals = [result['schemes'][n]['per_iter_total'] for n in names]
    x = range(len(names))
    width = 0.34
    gap = 0.03  # visual spacer between the paired bars

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    color_init, color_iter = '#2a78d6', '#eb6834'  # validated categorical slots 1-2
    bars_init = ax.bar([i - width / 2 - gap / 2 for i in x], init_vals, width,
                       label='Initialization (one-time)', color=color_init, zorder=3)
    bars_iter = ax.bar([i + width / 2 + gap / 2 for i in x], iter_vals, width,
                       label='Per training iteration', color=color_iter, zorder=3)

    ax.set_yscale('log')
    ax.set_ylabel('FLOPs (log scale)')
    ax.set_title('One-time initialization vs per-iteration training FLOPs')
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=18, ha='right', fontsize=8)
    ax.grid(axis='y', color='#d9d9d6', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='upper right')
    for bars in (bars_init, bars_iter):
        for b in bars:
            if b.get_height() == 0:
                continue
            ax.annotate(_human(b.get_height()), (b.get_x() + b.get_width() / 2,
                                                 b.get_height()),
                        textcoords='offset points', xytext=(0, 2),
                        ha='center', fontsize=6, color='#52514e')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = build_comparison(args)

    cfg = result['config']
    print(f"Config: L={cfg['depth']} d={cfg['in_dim']} d_h={cfg['hidden_dim']} "
          f"K={cfg['num_classes']} r={cfg['rank']} N={cfg['num_train_samples']} "
          f"B={cfg['batch_size']} opt={cfg['opt']} | LTH: p={cfg['lth_prune_rate']} "
          f"n={cfg['lth_rounds']} T={cfg['lth_iters_per_round']} "
          f"(final density {cfg['lth_final_density']:.4f}) | "
          f"DLRT rank={cfg['dlrt_rank']}\n")
    print_table(result)
    print_breakdowns(result)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nWrote {json_path}')
    if args.save_fig:
        fig_path = os.path.join(args.out_dir, 'flop_comparison.pdf')
        save_figure(result, fig_path)
        print(f'Wrote {fig_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
