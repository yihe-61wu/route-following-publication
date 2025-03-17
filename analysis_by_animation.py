import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import matplotlib.patches as mpatches
from utils import add_allo_ego


class route_following_animation:
    def __init__(self, data_dir, *models, animate_train=False):
        self.mb_thre = 0.8
        self.animate_train = animate_train
        self.dir = data_dir
        ls_model = list(models)
        ls_color = 'kb'
        self.ls_name = ['train'] + ls_model
        self.ls_file = ['Freight_train_1_record.npz'] + ['Freight_test_0_{}_record.npz'.format(m) for m in ls_model]
        self.fig, self.ax = plt.subplot_mosaic('''ab
                                                  ac''', figsize=(14, 14))
        self.x, self.y = {}, {}
        self.xh, self.yh = {}, {}
        self.xl, self.yl = {}, {}
        self.xr, self.yr = {}, {}
        self.xml, self.xmr, self.yml, self.ymr = {}, {}, {}, {}
        self.rz = {}
        self.mb_l, self.mb_r = {}, {}
        self.t_max = {}
        self.traj = {}
        self.path_h, self.path_l, self.path_r = {}, {}, {}
        self.ani_traj = {}
        self.head, self.view_l, self.view_r = {}, {}, {}
        self.nov_l, self.nov_r = {}, {}
        self.vl, self.v_lin = {}, {}
        xlim, ylim = [np.inf, -np.inf], [np.inf, -np.inf]
        for name, file, color in zip(self.ls_name, self.ls_file, ls_color):
            data_file = os.path.join(data_dir, file)
            data = np.load(data_file, allow_pickle='True')
            self.x[name], self.y[name], self.rz[name] = [data[key] for key in ('x', 'y', 'rz')]
            self.mb_l[name], self.mb_r[name], self.vl[name] = [data['extra'][()][key].flatten() for key in ('mb_l', 'mb_r', 'v_lin')]
            self.xl[name], self.yl[name] = add_allo_ego(self.x[name], self.y[name], self.rz[name] + np.pi / 4, self.mb_thre - self.mb_l[name])
            self.xr[name], self.yr[name] = add_allo_ego(self.x[name], self.y[name], self.rz[name] - np.pi / 4, self.mb_thre - self.mb_r[name])
            self.xml[name], self.yml[name], self.xmr[name], self.ymr[name] = [self.x[name][0]], [self.y[name][0]], [self.x[name][0]], [self.y[name][0]]
            for xxl, yyl, mml, xxr, yyr, mmr in zip(self.xl[name], self.yl[name], self.mb_l[name], self.xr[name], self.yr[name], self.mb_r[name]):
                if mml <= self.mb_thre:
                    self.xml[name].append(xxl)
                    self.yml[name].append(yyl)
                else:
                    self.xml[name].append(self.xml[name][-1])
                    self.yml[name].append(self.yml[name][-1])
                if mmr <= self.mb_thre:
                    self.xmr[name].append(xxr)
                    self.ymr[name].append(yyr)
                else:
                    self.xmr[name].append(self.xmr[name][-1])
                    self.ymr[name].append(self.ymr[name][-1])


            # self.xh[name], self.yh[name] = add_allo_ego(self.x[name], self.y[name], self.rz[name], self.mb_thre * 2 - self.mb_l[name] - self.mb_r[name])
            self.xh[name], self.yh[name] = (np.array(self.xml[name]) + np.array(self.xmr[name])) / 2, (np.array(self.yml[name]) + np.array(self.ymr[name])) / 2
            self.t_max[name] = len(self.x[name])
            self.traj[name], = self.ax['a'].plot([], [], color=color, label='{} path'.format(name), alpha=0.7)
            self.head[name], = self.ax['a'].plot([], [], color='tab:orange', ls='dashed')
            self.view_l[name], = self.ax['a'].plot([], [], color='g', ls='dashed')
            self.view_r[name], = self.ax['a'].plot([], [], color='r', ls='dashed')
            self.path_h[name], = self.ax['a'].plot([], [], color='tab:orange')#, ls='dotted', alpha=0.5)
            self.path_l[name], = self.ax['a'].plot([], [], color='g', ls='dotted', alpha=0.5)
            self.path_r[name], = self.ax['a'].plot([], [], color='r', ls='dotted', alpha=0.5)
            self.nov_l[name], = self.ax['b'].plot([], [], color='g')
            self.nov_r[name], = self.ax['b'].plot([], [], color='r')
            self.v_lin[name], = self.ax['c'].plot([], [], color='b')
            xlim[0] = np.min((xlim[0], self.x[name].min(), self.xh[name].min(), self.xl[name].min(), self.xr[name].min()))
            xlim[1] = np.max((xlim[1], self.x[name].max(), self.xh[name].max(), self.xl[name].max(), self.xr[name].max()))
            ylim[0] = np.min((ylim[0], self.y[name].min(), self.yh[name].min(), self.yl[name].min(), self.yr[name].min()))
            ylim[1] = np.max((ylim[1], self.y[name].max(), self.yh[name].max(), self.yl[name].max(), self.yr[name].max()))

        self.ax['a'].set_xlim(*xlim)
        self.ax['a'].set_ylim(*ylim)
        self.ax['a'].set_aspect('equal')
        self.ax['b'].set_xlim(0, np.max(list(self.t_max.values())))
        self.ax['c'].set_xlim(0, np.max(list(self.t_max.values())))
        self.ax['b'].set_ylim(0, 1)
        self.ax['c'].set_ylim(self.vl[name].min(), self.vl[name].max())
        if not self.animate_train:
            self.ax['a'].plot(self.x['train'], self.y['train'], color='k', alpha=0.7, label='train path')

        self.ax['b'].axhline(self.mb_thre, color='grey', alpha=0.7, linestyle='dotted', label='mb checkpoint threshold')
        self.ax['b'].set_title('mb output novelty')
        self.ax['c'].axhline(0.01, color='grey', alpha=0.7, linestyle='dotted', label='default speed for route recovery')
        self.ax['c'].set_title('linear speed')
        for ak in 'abc':
            self.ax[ak].legend()

    def _update_trajectory(self, frame, name):
        self.traj[name].set_data(self.x[name][:frame + 1], self.y[name][:frame + 1])
        self.path_h[name].set_data(self.xh[name][:frame + 1], self.yh[name][:frame + 1])
        self.path_l[name].set_data(self.xl[name][:frame + 1], self.yl[name][:frame + 1])
        self.path_r[name].set_data(self.xr[name][:frame + 1], self.yr[name][:frame + 1])
        x0, y0 = self.x[name][frame], self.y[name][frame]
        xh, yh = self.xh[name][frame], self.yh[name][frame]
        xl, yl = self.xml[name][frame], self.yml[name][frame]
        xr, yr = self.xmr[name][frame], self.ymr[name][frame]
        self.head[name].set_data([x0, xh], [y0, yh])
        self.view_l[name].set_data([x0, xl], [y0, yl])
        self.view_r[name].set_data([x0, xr], [y0, yr])
        self.nov_l[name].set_data(np.arange(frame + 1), self.mb_l[name][:frame + 1])
        self.nov_r[name].set_data(np.arange(frame + 1), self.mb_r[name][:frame + 1])
        self.v_lin[name].set_data(np.arange(frame + 1), self.vl[name][:frame + 1])
        return self.traj[name], self.head[name], self.view_l[name], self.view_r[name], self.path_h[name], self.path_l[name], self.path_r[name], self.nov_l[name], self.nov_r[name], self.v_lin[name]

    def animate_trajectory(self):
        for name in self.ls_name:
            if name == 'train' and not self.animate_train: continue
            self.ani_traj[name] = ani.FuncAnimation(fig=self.fig, func=self._update_trajectory, fargs=(name,),
                                                    frames=self.t_max[name], interval=33, repeat=True)
            self.ani_traj[name].save(filename=os.path.join(self.dir, '{}.mp4'.format(name)))


if __name__ == "__main__":
    # data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/test1/sim_20240517_165554_random'
    # data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/test_learn_middle/sim_20240520_104919_random' # sim_20240520_110034_random'
    # data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/test2/sim_20240520_130915_random'
    data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/checkpoint-success/sim_20240520_142705_random' #sim_20240520_151415_random' # -- success turning at sharp!

    models = 'dimb-ST',
    rfa = route_following_animation(data_dir, *models, animate_train=False)
    ani = rfa.animate_trajectory()
    plt.show()