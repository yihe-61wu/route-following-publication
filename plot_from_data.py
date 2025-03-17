import matplotlib.pyplot as plt
import numpy as np
import os
from analysis_by_plot import draw_floorplan


class DataVisualiser:
    def __init__(self, data_dir, headless=False):
        self.headless = headless
        self.data_dir = data_dir
        self.data_path = []
        for data_name in os.listdir(data_dir):
            if data_name.endswith('_record.npz'): self.data_path.append(os.path.join(data_dir, data_name))
        data = np.load(self.data_path[0], allow_pickle=True)
        self.var_major = list(data.keys())
        self.var_extra = list(data['extra'][()].keys())
        if not headless: print(self.var_major, self.var_extra)

    def _complete_draw(self, figname, xlabel=[], ylabel=[], grid='on', legendloc=0):
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(grid)
        plt.legend(loc=legendloc)
        if self.headless: plt.savefig(os.path.join(self.data_dir, figname))

    def draw_summary(self, list_var=[]):
        self.draw_trajectory()
        [self.draw_temporal_profile(var) for var in list_var]
        if self.headless:
            plt.close('all')
        else:
            plt.show()

    def draw_temporal_profile(self, var, figname=None):
        if figname is None: figname = var
        plt.figure(figname)
        for data_file in self.data_path:
            data = np.load(data_file, allow_pickle=True)
            if var in self.var_extra:
                val = data['extra'][()][var]
            else:
                val = data[var]
            take_idx, robot_name = [data[var] for var in ('take_idx', 'robot_name')]
            label = "{}-{}".format(robot_name, take_idx)
            plt.plot(val, label=label)

        self._complete_draw(figname, 'epoch', var)

    def draw_trajectory(self):
        figname = 'trajectory'
        plt.figure(figname)

        for data_file in np.sort(self.data_path):
            data = np.load(data_file, allow_pickle=True)
            x, y, take_idx, robot_name, scene_name = [data[var] for var in ('x', 'y', 'take_idx', 'robot_name', 'scene_name')]
            label = "{}-{}".format(robot_name, take_idx)
            lines = plt.plot(x, y, label=label)
            if np.all([var in data['extra'][()] for var in ('collision', 'proximity', 'normal')]):
                collision, proximity, normal = [data['extra'][()][var] for var in ('collision', 'proximity', 'normal')]
                linecolor = lines[0].get_color()
                plt.scatter(np.where(normal, x, np.nan), np.where(normal, y, np.nan), marker='.')
                plt.scatter(np.where(proximity, x, np.nan), np.where(proximity, y, np.nan), marker='s', facecolor='none', edgecolor=linecolor)
                plt.scatter(np.where(collision, x, np.nan), np.where(collision, y, np.nan), marker='X', facecolor='none', edgecolor=linecolor)

        ax = plt.gca()
        ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        floorplan = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/data/ig_dataset/scenes/{}/layout/floor_trav_no_door_0.png'.format(scene_name)
        draw_floorplan(ax, floorplan, 0.01)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        self._complete_draw(figname, 'x', 'y')


if __name__ == '__main__':
    data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/test_goal_directed/sim_20240702_123432_Pomaria_1_int'
    dv = DataVisualiser(data_dir, headless=False)
    dv.draw_trajectory()
    # dv.draw_temporal_profile('vl', 'oracle')
    # dv.draw_temporal_profile('v_lin', 'command')
    plt.show()