import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def draw_floorplan(ax, floorplan, map_resolution, k_rotation=0):
    img = plt.imread(floorplan)
    xylim = img.shape[0] / 2 * map_resolution
    im = ax.imshow(np.rot90(img, k=k_rotation), cmap='gray', origin='lower', extent=(-xylim, xylim, -xylim, xylim))
    return im

def myboxplot(ax, data, labels, colors, alpha_points, size_points, vertical=True):
    # plot individual data
    xticks = np.arange(len(data)) + 1
    for xtick, group_data in zip(xticks, data):
        if vertical:
            x, y = np.random.normal(xtick, 0.04, len(group_data)), group_data
        else:
            y, x = np.random.normal(xtick, 0.04, len(group_data)), group_data
        ax.scatter(x, y, facecolor='none', edgecolor='k', alpha=alpha_points, s=size_points)
    # boxplot
    bp = ax.boxplot(data, labels=labels, vert=vertical,
                    showfliers=False, showmeans=True, meanline=True, patch_artist=True, widths=0.8)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for linekey in ('means', 'medians'):
        for line, color in zip(bp[linekey], colors):
            line.set_color(color)
    return ax

def pheatmap(ax, data, vmin=-10, vmax=0):
    matp = group_pairwise_test(data)
    matlogp = np.log10(matp, where=matp > 0, out=np.full_like(matp, -np.inf))
    matkorw = np.where(matlogp < (vmin - vmax) / 2, 'w', 'k')
    im = ax.imshow(matlogp.astype(int), cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
    for i in range(matp.shape[0]):
        for j in range(matp.shape[1]):
            if i != j: ax.text(j, i, np.where(matlogp[i, j] == -np.inf, '$\infty$', np.round(-matlogp[i, j], 0).astype(int)),
                               ha="center", va="center", color=matkorw[i, j], fontsize=6)
    return im

def myttests(data, comparison_indices, ispaired, alternative='two-sided', permutations=0):
    if ispaired:
        t_p = lambda y1, y0: stats.ttest_rel(y1, y0, alternative=alternative).pvalue
    else:
        t_p = lambda y1, y0: stats.ttest_ind(y1, y0, alternative=alternative, permutations=permutations).pvalue
    pvalues = []
    for idx1, idx0 in comparison_indices:
        t_pvalue = t_p(data[idx1], data[idx0])
        pvalues.append(t_pvalue)
    return pvalues

def compare_ind_group(x, y, alternative='two-sided', t_equalvar=True, t_permutations=None):
    Ttest_result = stats.ttest_ind(x, y, alternative=alternative, equal_var=t_equalvar, permutations=t_permutations)
    Mrank_result = stats.mannwhitneyu(x, y, alternative=alternative)
    pvals = Ttest_result.pvalue, Mrank_result.pvalue
    return pvals

def group_pairwise_test(data, equalvar=False, permutations=100000):
    mat_pval = np.identity(len(data))
    for idx, x in enumerate(data[:-1]):
        for jdx in np.arange(idx + 1, len(data)):
            mat_pval[idx, jdx], mat_pval[jdx, idx] = compare_ind_group(x, data[jdx],
                                                                t_equalvar=equalvar, t_permutations=permutations)
    return mat_pval

def plot2metric(data_goaldist, data_pathdsim,
                ls_model_name, ls_model_abbr, ls_color, dot_alpha, dot_size,
                figsize, dpi=300):
    fig, axes = plt.subplot_mosaic('''
                                    AAC
                                    BBD
                                    ''', figsize=figsize, dpi=dpi)
    fig.subplots_adjust(hspace=0.05, wspace=0)

    for ak1, ak2, data, xlabel in zip('AB',
                                      'CD',
                                      (data_goaldist, data_pathdsim),
                                      ('goal distance $\delta_g$ (m)', 'path dissimilarity $\delta_\Gamma$ (m)')
                                      ):
        ax1, ax2 = axes[ak1], axes[ak2]
        # boxplot
        myboxplot(ax1, data, ls_model_name, ls_color, dot_alpha, dot_size, vertical=False)
        ax1.set_xlabel(xlabel, fontsize=6)
        ax1.xaxis.set_tick_params(labelsize=6)
        ax1.yaxis.set_tick_params(labelsize=6)
        ax1.grid()
        # heatmap
        im = pheatmap(ax2, data)
        ax2.set_xticks(np.arange(len(ls_model_name)))
        ax2.set_yticks([])

    axes['A'].xaxis.set_label_position('top')
    axes['A'].xaxis.set_ticks_position('top')
    axes['C'].set_title('significance', fontsize=6)
    axes['D'].set_xlabel('$-$log$_{10}$($p$) of U/t', fontsize=6)
    axes['D'].set_xticklabels(ls_model_abbr, fontsize=6)
    cbar = fig.colorbar(im, ax=[axes['C'], axes['D']], ticks=(0,-2,-10))
    cbar.ax.set_yticklabels([0, 2, '$\geq 10$'])
    cbar.ax.tick_params(labelsize=6)

def familiarity_polar(centre, data360):
    _t = np.arange(0, 360) / 180 * np.pi
    cx, cy = np.cos(_t), np.sin(_t)
    _x = np.multiply(cx, data360)
    _y = np.multiply(cy, data360)
    return centre[0] + _x, centre[1] + _y



def familiarity_summary(data_dir, var_names=[], save_dir=None, headless=True):
    if save_dir is None:
        save_dir = data_dir

    fig_name = 'familiarity along route'
    plt.figure(fig_name)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    data_train =  np.load(os.path.join(data_dir, 'Freight_train_0_record.npz'), allow_pickle=True)
    x, y = data_train['x'], data_train['y']
    plt.plot(-y, x, label='training route', color='k')

    data_test =  np.load(os.path.join(data_dir, 'Freight_test_0_record.npz'), allow_pickle=True)
    x, y = data_test['x'], data_test['y']
    plt.plot(-y, x, label='sample test trajectory', color='g', marker='o', alpha=0.5)

    ls_var = 'robot_name', 'take_idx', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'extra'
    for session_idx, data_file in enumerate(os.listdir(data_dir)):
    # for Nview, a, m, c in zip((1, 5, 25, 125), (.55, .7, .85, 1), ('solid', 'dotted', '-.', 'dashed'), 'rgbk'):
    #     data_file = 'Freight_analysis_{}_record.npz'.format(Nview)
        if data_file.endswith('_record.npz') and data_file.startswith('Freight_analysis_'):
            data_path = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
            robot_name, take_idx, x, y, z, rx, ry, rz, extra = [data_path[var] for var in ls_var]
            label = "{}-{}".format(robot_name, take_idx)


            for key in var_names:
                figtit = '{}'.format(key)
                plt.figure(figtit)
                plt.xlabel('epoch')
                fam2 = np.concatenate((extra[()][key][180:], extra[()][key][:-180]))
                plt.plot((fam2 - np.min(fam2)) / (np.max(fam2) - np.min(fam2)),
                         # label='{} view(s) learned'.format(Nview),
                         # c=c, ls=m)
                         )

                # plt.grid('on')
                # plt.legend(loc=0)
                # plt.savefig(os.path.join(save_dir, figtit))

            plt.figure(fig_name)
            fam_r, fam_l = extra[()]['fam_r'], extra[()]['fam_l']
            rot = np.where(fam_r + fam_l <= 0, np.sign(fam_r - fam_l), (fam_r - fam_l) / (fam_r + fam_l)) / 2
            ridx = np.argwhere(rot >= 0)
            lidx = np.argwhere(rot < 0)
            rrot, lrot = np.abs(rot), np.abs(rot)
            rrot[lidx] = np.nan
            lrot[ridx] = np.nan
            rx, ry = familiarity_polar([x[0], y[0]], rrot)
            lx, ly = familiarity_polar([x[0], y[0]], lrot)
            plt.plot(-ry, rx, color='r', ls='dotted')
            plt.plot(-ly, lx, color='b', ls='dashed')
            # fx, fy = familiarity_polar([x[0], y[0]], (fam_r + fam_l) / 50)
            # plt.plot(-fy, fx, color='g')

    plt.figure(fig_name)
    plt.plot([], color='r', ls='dotted', label='rightward angle (a.u.)')
    plt.plot([], color='b', ls='dashed', label='leftward angle (a.u.)')
    # plt.plot([], color='g', label='familiarity')
    plt.grid('on')
    plt.legend(loc=4)
    plt.gca().set_aspect('equal')
    plt.savefig(os.path.join(save_dir, fig_name))
    plt.xticks([-3, -2, -1, 0, 1, 2, 3])
    plt.yticks([-1, 0, 1])
    plt.title('sample trajectory and rotation analysis')


    plt.figure(figtit)
    plt.axvline(180, color='k')
    plt.xticks(np.arange(0, 361, 60), np.arange(-180, 181, 60))
    plt.legend()
    plt.grid()
    plt.ylabel(r'normalised similarity $= 1 -$ normalised distance')
    plt.xlabel('angular offset from correct orientation (deg)')
    plt.title('left familiarity (FlyHash)') # should learn opposite



    if not headless:
        plt.show()

    plt.close('all')


if __name__ == '__main__':
    # data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/mbtest/mbvaryfreq'
    # ####data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/rotate_data/mb-varylearningfreq'
    data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/rotate_data/sim_20230516_143535_Rs_int'
    familiarity_summary(data_dir, ['fam_r'], headless=False)