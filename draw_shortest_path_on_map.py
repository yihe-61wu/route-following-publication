import logging
import os
from datetime import datetime
from sys import platform

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import cv2

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
from igibson.utils.assets_utils import get_ig_scene_path
from igibson.utils.transform_utils import quat2axisangle, axisangle2quat

from video_recorder import Recorder
from robot_main import *

from igibson_api import NoisyDDController, SimulationOrganisor

from plot_from_data import DataVisualiser
from utils import pairwise_distances


def run_lamb(N_kc=32000, N_pn_kc=10, N_memory=None, sparsity=0.1, visual_noise=0, motor_noise=0, rot_coeff=10,
         save_dir='test', rand_test_init_orn=False, headless=True):
    scene_name = 'random'
    robot_name = 'freight'
    simulation = SimulationOrganisor(scene_name, robot_name, headless)
    robot = simulation.prepare_robot()
    scene = simulation.prepare_scene(load_object_categories=None, trav_map_resolution=0.01)
    s = simulation.prepare_simulator()

    # s.import_scene(scene)
    # s.import_object(robot)
    simulation.initialise_simulation()
    recorder = Recorder(simulation, save_dir)

    pts_abc = [scene.get_random_point()[1][:2] for _ in range(3)]
    print(pts_abc)
    dist_abc = pairwise_distances(pts_abc, pts_abc)
    dist_ab_bc_ca = dist_abc[1, 0], dist_abc[2, 1], dist_abc[0, 2]
    print(dist_ab_bc_ca)
    idx_0, idx_1 = np.argsort(dist_ab_bc_ca)[:2]
    idx_max = np.argmax(dist_ab_bc_ca)

    path = np.empty((0, 2))
    for i in (0, 1):
        start = pts_abc[idx_max - i]
        end = pts_abc[idx_max - i - 1]

    # start = scene.get_random_point_by_room_type('bathroom')[1][:2]
    # end = scene.get_random_point_by_room_type('kitchen')[1][:2]

        print('world:', start, end)
        print('map:', scene.world_to_map(start), scene.world_to_map(end))
        shortest_path = scene.get_shortest_path(0, start, end, True)
        print('path length:', shortest_path[1])
        path = np.concatenate((path, shortest_path[0]), axis=0)

    print(path.shape)
    spl = scp.interpolate.make_interp_spline(np.arange(path.shape[0]), path)
    path = spl(np.linspace(0, path.shape[0], 120))
    print(path.shape)



    # training
    recorder.start_recording()
    train_time = path.shape[0]
    print(train_time)
    for t_ctrl in range(train_time):
        t_next = t_ctrl + 1
        pos = path[t_ctrl]
        if t_ctrl < train_time - 1:
            pos_next = path[t_next]
            orn_vec = pos_next - pos
            orn_rad = np.arctan2(orn_vec[1], orn_vec[0])
            orn_quat = axisangle2quat([0, 0, orn_rad])
        robot.set_position_orientation(np.concatenate((pos, [0])), orn_quat)

        s.step()
        recorder.recording()#orn=orn_rad)

    take_idx = 'random_train'
    recorder.stop_recording(take_idx)
    s.disconnect()


    # to be integrated in plot analysis
    # draw path
    floorplan = os.path.join(scene.scene_dir, 'layout', 'floor_trav_no_door_0.png')
    fig, ax = plt.subplots()
    img = plt.imread(floorplan)
    scale_factor = scene.trav_map_default_resolution / scene.trav_map_resolution
    img_resize = cv2.resize(img, np.multiply(img.shape, scale_factor).astype(int))
    ax.imshow(img_resize, cmap='gray')

    sp = scene.world_to_map(path).T
    ax.plot(sp[1], sp[0], c='r', marker='o')

    robot_wheel_size = robot.wheel_axle_length
    sizebar = scene.world_to_map([[0, robot_wheel_size], [0, 0], [robot_wheel_size, 0]]) + [1, 1]
    ax.plot(*sizebar.T, lw=2, c='b')

    plt.show()


def find_shortest_path(scene_name, start, goal):
    simulation = SimulationOrganisor(scene_name, 'freight', True)
    robot = simulation.prepare_robot()
    scene = simulation.prepare_scene(load_object_categories=None, trav_map_resolution=0.01)
    s = simulation.prepare_simulator()
    s.import_scene(scene)
    shortest_path = scene.get_shortest_path(0, start[:2], goal[:2], True)
    return shortest_path


if __name__ == "__main__":
    # save_dir = 'test'
    # for _ in range(1):
    #     run_lamb(save_dir=save_dir, headless=False)

    scene_name = 'Beechwood_0_int'
    start, goal = [-0.69, -3.45], [-8.34,  4.68]
    print(find_shortest_path(scene_name, start, goal))