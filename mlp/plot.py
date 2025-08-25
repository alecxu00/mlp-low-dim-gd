import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Union
import os


def plot_train_losses(train_losses: Union[List, np.array],
                      act_fn_str: str, 
                      d: int, 
                      p: int,
                      K: int, 
                      save_fname: str, 
                      savefig: bool = False):
    
    
    title_ = act_fn_str + ' Network Train Loss (d = ' + str(d) + ', K = ' + str(K) + ')'
    plt.figure()
    if isinstance(train_losses, np.ndarray) and len(train_losses.shape) > 1:
        mean_train_losses = np.mean(train_losses, axis=0)
        std_train_losses = np.std(train_losses, axis=0)
        
        plt.plot(np.arange(len(mean_train_losses)) + 1, mean_train_losses)
        plt.fill_between(np.arange(len(mean_train_losses)) + 1, mean_train_losses - std_train_losses, mean_train_losses+ std_train_losses, alpha=0.5)
    
    else:
        plt.plot(np.arange(len(train_losses)) + 1, train_losses)
    
    plt.title(title_)
    plt.xlabel('Iteration (t)')
    plt.ylabel('Train Loss')
    
    if savefig:
        plt.savefig(save_fname)
    plt.show()
    plt.close()

def plot_series(
    ax,
    series,
    layer=0,
    color="viridis",
    crossing_lines=False,
    x_points=None,
    y_points=None,
    zoom=0.8,
    elev=30,
    azim=-50,
    roll=0,
    alpha=1.0,
    linewidth=1.0,
):
    n_x_indices, n_y_indices = series.shape[0], series.shape[-1]
    x_indices = np.arange(n_x_indices) + 1
    y_indices = np.arange(n_y_indices) + 1

    if x_points is None:
        x_points = x_indices
    if y_points is None:
        y_points = y_indices

    spectrum_verts = []

    for idx in x_indices:
        spectrum_verts.append(
            [
                (0, np.min(series) - 0.1),
                *zip(y_points, series[idx - 1, layer, :]),
                (y_points[-1], np.min(series) - 0.1),
            ]
        )

    spectrum_poly = PolyCollection(spectrum_verts)
    spectrum_poly.set_linewidth(linewidth)
    spectrum_poly.set_alpha(alpha)
    spectrum_poly.set_facecolor(
        plt.colormaps[color](np.linspace(0, 0.7, len(spectrum_verts)))  # type: ignore
    )
    spectrum_poly.set_edgecolor("black")

    ax.set_box_aspect(aspect=None, zoom=zoom)

    ax.add_collection3d(spectrum_poly, zs=x_indices, zdir="y")

    if crossing_lines:
        path_verts = []

        for idx in y_indices:
            path_verts.append([*zip(x_points, series[:, layer, idx - 1])])
        path_line = LineCollection(path_verts)
        path_line.set_linewidth(linewidth)
        path_line.set_edgecolor("black")
        ax.add_collection3d(path_line, zs=y_indices, zdir="x")

    ax.set_xlim(y_points[0], y_points[-1])
    ax.set_ylim(x_points[0], x_points[-1])
    ax.set_zlim(np.min(series) - 0.1, np.max(series) + 0.1)

    ax.view_init(elev, azim, roll)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_weight_svd_angles(angles_dict: Dict[str, np.ndarray],
                           act_str: str, 
                           d: int, 
                           p: int, 
                           K: int, 
                           save_dir: str,
                           savefig: bool = False):
    L = angles_dict['svals'].shape[1]
    for l in range(L):
        fig = plt.figure(figsize=(21, 7))
        
        ax1 = fig.add_subplot(131, projection='3d')
        plot_series(ax1, angles_dict['svals'], layer=l, crossing_lines=True, linewidth=0.5, alpha=1.0)
        
        ax1.set_xlabel('Singular Value Index (i)', fontsize=12)
        ax1.set_ylabel('Iteration (t)', fontsize=12)
        ax1.set_zlabel(r'$\sigma_i(t)$', fontsize=12)

        ax1.set_title('Singular Values', fontsize=18, y=0.0)

        ax1.xaxis.set_tick_params(labelsize=8)
        ax1.yaxis.set_tick_params(labelsize=8)
        ax1.zaxis.set_tick_params(labelsize=8)

        
        ax2 = fig.add_subplot(132, projection='3d')
        plot_series(ax2, angles_dict['U_all_angles'], layer=l, crossing_lines=True, linewidth=0.5, alpha=1.0)

        ax2.set_xlabel('Singular Vector Index (i)', fontsize=12)
        ax2.set_ylabel('Iteration (t)', fontsize=12)
        ax2.set_zlabel(r'$\angle u_i(t), u_i(0)$ (Rad.)', fontsize=12)

        ax2.set_title('Left Singular Vectors', fontsize=18, y=0.0)

        ax2.xaxis.set_tick_params(labelsize=8)
        ax2.yaxis.set_tick_params(labelsize=8)
        ax2.zaxis.set_tick_params(labelsize=8)


        ax3 = fig.add_subplot(133, projection='3d')
        plot_series(ax3, angles_dict['V_all_angles'], layer=l, crossing_lines=True, linewidth=0.5, alpha=1.0)

        ax3.set_xlabel('Singular Vector Index (i)', fontsize=12)
        ax3.set_ylabel('Iteration (t)', fontsize=12)
        ax3.set_zlabel(r'$\angle v_i(t), v_i(0)$ (Rad.)', fontsize=12)

        ax3.set_title('Right Singular Vectors', fontsize=18, y=0.0)

        ax3.xaxis.set_tick_params(labelsize=8)
        ax3.yaxis.set_tick_params(labelsize=8)
        ax3.zaxis.set_tick_params(labelsize=8)

        plt.suptitle(act_str + r' Network $W_' + str(l + 1) + '$ SVD Components (d = ' + str(d) + ', K = ' + str(K) + ')', fontsize=24)

        plt.tight_layout()
        save_fname = os.path.join(save_dir, 'layer_' + str(l + 1) + 'svd_angles.pdf')
        if savefig:
            plt.savefig(save_fname)
        plt.show()
        plt.close()
    

def plot_weight_svd_errors(errors_dict: Dict[str, np.ndarray], 
                           act_str: str, 
                           d: int, 
                           p: int, 
                           K: int, 
                           save_dir: str,
                           subspace_type: int = 1,
                           data_type: str = 'Structured', 
                           savefig: bool = False):
    svals = errors_dict['svals']

    U_top_errors = errors_dict["U_top_errors"]
    U_mid_errors = errors_dict["U_mid_errors"]
    U_bot_errors = errors_dict["U_bot_errors"]

    V_top_errors = errors_dict["V_top_errors"]
    V_mid_errors = errors_dict["V_mid_errors"]
    V_bot_errors = errors_dict["V_bot_errors"]

    epochs, L = U_top_errors.shape

    for l in range(L):
        fig = plt.figure(figsize=(12, 5))
        
        # Singular values
        ax1 = fig.add_subplot(131, projection='3d')
        plot_series(ax1, svals, layer=l, crossing_lines=True, linewidth=0.5, alpha=1.0)
        
        ax1.set_xlabel(r'Singular Value Index ($i$)', fontsize=10)
        ax1.set_ylabel(r'Epoch ($t$)', fontsize=10)
        ax1.set_zlabel(r'$\sigma_i(t)$', fontsize=10)

        ax1.set_title('Singular Values', fontsize=12)
        ax1.xaxis.set_tick_params(labelsize=6)
        ax1.yaxis.set_tick_params(labelsize=6)
        ax1.zaxis.set_tick_params(labelsize=6)
        
        # Left singular subspaces
        ax2 = fig.add_subplot(132)
        ax2.plot(np.arange(epochs) + 1, U_top_errors[:, l], label='Top')
        ax2.plot(np.arange(epochs) + 1, U_mid_errors[:, l], label='Middle')
        ax2.plot(np.arange(epochs) + 1, U_bot_errors[:, l], label='Bottom')
        ax2.set_title('Left Singular Subspaces', fontsize=12)
        ax2.set_ylabel('Subspace Error', fontsize=12)
        ax2.set_yscale('log')
        ax2.grid()
        
        
        # Right singular subspaces
        ax3 = fig.add_subplot(133, sharey=ax2)
        ax3.plot(np.arange(epochs) + 1, V_top_errors[:, l], label='Top')
        ax3.plot(np.arange(epochs) + 1, V_mid_errors[:, l], label='Middle')
        ax3.plot(np.arange(epochs) + 1, V_bot_errors[:, l], label='Bottom')
        ax3.set_title('Right Singular Subspaces', fontsize=12)
        ax3.legend(loc='lower right')
        ax3.set_yscale('log')
        ax3.grid()
        
        # Figure 
        suptitle = act_str + r' Network $W_' + str(l + 1) + r'$ Subspace Errors (' + data_type + ' Data, d = ' + str(d) + ', K = ' + str(K) + ')' #, Subspace ' + str(subspace_type)
#         if subspace_type == 1:
#             suptitle = suptitle + r', Subspace $S$)'
#         elif subspace_type == 2:
#             suptitle = suptitle + r', Subspace $\widetilde{S}$)'
#         else:
#             raise ValueError('Invalid subspace type.')
        
        fig.suptitle(suptitle, fontsize=15)
        fig.supxlabel(r'Epoch ($t$)', x=2/3, y=0.1)
        #fig.supylabel('Subspace Error')

        plt.tight_layout(pad=2.0)

        save_fname = os.path.join(save_dir, 'layer_' + str(l + 1) + '_weight_svd_errors_subspace_' + str(subspace_type) + '.pdf')
        if savefig:
            plt.savefig(save_fname)
        plt.show()
        plt.close()
