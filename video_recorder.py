import os
from PIL import Image
import subprocess

import numpy as np

import cv2
import matplotlib.pyplot as plt

from igibson_api import SimulationOrganisor
from robot_main import EmbodiedSensoryMotorModel


class Recorder:
    def __init__(self,
                 simulation_info: SimulationOrganisor,
                 save_dir,
                 *extra_var):
        if save_dir is None:
            self.save_dir = os.path.join('records', 'sim' + simulation_info.stamp)
        else:
            self.save_dir = os.path.join('records', save_dir, 'sim' + simulation_info.stamp)
        os.makedirs(self.save_dir, exist_ok=True)

        self.oracle = simulation_info.oracle
        self.scene_name = simulation_info.scene_name
        self.robot_name = simulation_info.robot_name
        self.simulator = simulation_info.simulator
        self.headless = simulation_info.headless
        self.pos_start = simulation_info.pos_start
        self.pos_goal = simulation_info.pos_goal
        self.optimal_route = simulation_info.optimal_route

        self.extra_data = {}
        self.extra_data_name = extra_var

    def start_recording(self):
        self.robot_pos = []
        self.robot_orn = []
        self.robot_vl = []
        self.robot_va = []
        self.robot_force = []
        [self.extra_data.update({edn: []}) for edn in self.extra_data_name]

        if not self.headless:
            self.record_time = 0
            self.tmp_robot_dir = os.path.join(self.save_dir, "tmp_robot_" + self.robot_name)
            os.makedirs(self.tmp_robot_dir, exist_ok=True)
            self.tmp_bird_dir = os.path.join(self.save_dir, "tmp_bird_" + self.robot_name)
            os.makedirs(self.tmp_bird_dir, exist_ok=True)

    def recording(self, **kwargs):
        robot_pos = self.oracle.get_position(verbose=True)
        robot_orn = self.oracle.get_orientation(verbose=True)
        robot_vl = self.oracle.get_speed_linear()
        robot_va = self.oracle.get_speed_angular()
        robot_force = self.oracle.get_force_horizon()
        self.robot_pos.append(robot_pos)
        self.robot_orn.append(robot_orn)
        self.robot_vl.append(robot_vl)
        self.robot_va.append(robot_va)
        self.robot_force.append(robot_force)
        [self.extra_data[edn].append(kwargs[edn]) for edn in self.extra_data_name]

        if not self.headless:
            frames_roboteye = self.oracle.get_frame_rgb()

            bird = self.simulator.viewer
            bird.px, bird.py, bird.pz = robot_pos - bird.initial_view_direction * 2
            bird_position = np.array([bird.px, bird.py, bird.pz])
            bird.renderer.set_camera(bird_position, bird_position + bird.view_direction, bird.up)
            frames_birdeye = bird.renderer.render(modes=("rgb"))[0][:, :, :3]
            for frame, tmp_dir in zip((frames_roboteye, frames_birdeye), (self.tmp_robot_dir, self.tmp_bird_dir)):
                img = Image.fromarray((255 * frame).astype(np.uint8))
                img.save(os.path.join(tmp_dir, "{:05d}.png".format(self.record_time)))

            self.record_time += 1

    def stop_recording(self, take_idx, verbose=False):
        robot_pos = np.transpose(self.robot_pos)
        robot_orn = np.transpose(self.robot_orn)
        extra = {}
        [extra.update({edn: np.transpose(self.extra_data[edn])}) for edn in self.extra_data_name]

        data_path = os.path.join(self.save_dir, "{}_{}_record".format(self.robot_name, take_idx))
        np.savez(data_path,
                 x=robot_pos[0],
                 y=robot_pos[1],
                 z=robot_pos[2],
                 rx=robot_orn[0],
                 ry=robot_orn[1],
                 rz=robot_orn[2],
                 vl=self.robot_vl,
                 va=self.robot_va,
                 f=self.robot_force,
                 extra=extra,   # dictionary to be loaded by ['extra'][()]
                 robot_name=self.robot_name,
                 scene_name=self.scene_name,
                 pos_start=self.pos_start,
                 pos_goal=self.pos_goal,
                 optimal_route=self.optimal_route,
                 take_idx=take_idx)

        if not self.headless:
            for observer, tmp_dir in zip(('robot', 'bird'), (self.tmp_robot_dir, self.tmp_bird_dir)):
                cmd = "ffmpeg -i {t}/%5d.png -y -an -c:v libx264 -crf 18 -preset veryslow -r 30 -pix_fmt yuv420p {s}/{n}_{o}_eye_{take}.mp4".format(
                    t=tmp_dir,
                    s=self.save_dir,
                    o=observer,
                    n=self.robot_name,
                    take=take_idx
                )
                subprocess.call(cmd, shell=True)
                cmd = "rm -r {}".format(tmp_dir)
                subprocess.call(cmd, shell=True)


# to be deprecated, replaced by DataVisualiser in plot_from_data.py
def plot_summary(data_dir, var_names=[], save_dir=None, headless=True):
    if save_dir is None:
        save_dir = data_dir

    plt.figure('locomotion')
    plt.xlabel('x')
    plt.ylabel('y')

    ls_var = 'robot_name', 'take_idx', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'extra'
    for session_idx, data_file in enumerate(os.listdir(data_dir)):
        if data_file.endswith('_record.npz'):
            data_path = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
            robot_name, take_idx, x, y, z, rx, ry, rz, extra = [data_path[var] for var in ls_var]
            label = "{}-{}".format(robot_name, take_idx)

            plt.figure('locomotion')
            plt.plot(-y, x, label=label, marker='.')
            plt.grid('on')
            plt.legend(loc=0)
            plt.savefig(os.path.join(save_dir, 'locomotion'))

            for key in var_names:
                figtit = '{}'.format(key)
                plt.figure(figtit)
                plt.xlabel('epoch')
                plt.plot(extra[()][key], label=label + '-' + key)

                plt.grid('on')
                plt.legend(loc=0)
                plt.savefig(os.path.join(save_dir, figtit))

    if not headless:
        plt.show()

    plt.close('all')


if __name__ == '__main__':
    data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/vary_Npnkc_500epoch/N_pnkc600/sim_20230320_131259_Rs_int'
    plot_summary(data_dir, ['v_lin', 'v_ang'], headless=False)