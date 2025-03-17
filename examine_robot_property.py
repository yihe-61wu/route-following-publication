import logging
import os.path

import numpy as np
from simulation_main import *
from video_recorder import Recorder
from plot_from_data import DataVisualiser
from igibson.utils.transform_utils import axisangle2quat


def main(model_name='lamb',
         scene_name='empty',
         save_dir='test',
         headless=True,
         **kwargs):
    ### initialisation
    for key, value in kwargs.items():
        kwargs_main[key] = value
    kwargs_main['obstacle_avoidance'] = False

    ####
    # erosion size is super important for finding shortest path (although this might not be needed if real training available)
    # erosion should be found via footprint rather than wheel_axlenght

    # locobot and turtlebot indeed have the same WHEEL
    for robot_name in ('freight', 'locobot', 'turtlebot'):
        simulation = SimulationOrganisor(scene_name, robot_name, headless)
        robot = simulation.prepare_robot()
        scene = simulation.prepare_scene()
        s = simulation.prepare_simulator()
        simulation.initialise_simulation()

        robot_maxwheelspeed = robot.control_limits['velocity'][1][1]
        robot_maxva = 2 * robot.wheel_radius * robot_maxwheelspeed / robot.wheel_axle_length
        robot_maxvl = robot.wheel_radius * robot_maxwheelspeed
        print(robot_name, robot_maxva, robot_maxvl)

    # init_pose = simulation.init_robot_pos_orn
    # toy_brain = construct_model(model_name, simulation, **kwargs_main)
    # var_monitor = ['collision', 'proximity', 'normal',
    #                'v_ang', 'v_ang_true']
    # recorder = Recorder(simulation, save_dir, *var_monitor)



    # ### training
    # N_train = 1
    # print('start training ...')
    # for trial in range(N_train):
    #     print('training trial {}'.format(trial))
    #     simulation.reinit_robot_pose(True)
    #     orn_goal = simulation.oracle.get_orientation()
    #     recorder.start_recording()
    #
    #     for _ in range(60):
    #         _, (v_lin, v_ang), perception, collided, proximal, normal = toy_brain.control(True, (0, 1.1), noisefree=False)
    #         s.step()
    #         v_ang_true = simulation.oracle.get_speed_angular()
    #
    #         recorder.recording(v_ang=v_ang, v_ang_true=v_ang_true,
    #                            collision=collided, proximity=proximal, normal=normal)
    #
    #     robot.keep_still()
    #     take_idx = 'train_{}'.format(trial)
    #     recorder.stop_recording(take_idx)
    #
    # s.disconnect()
    #
    # var_names = ['v_ang', 'v_ang_true']
    # # var_names = []
    # painter = DataVisualiser(recorder.save_dir, headless)
    # painter.draw_summary(var_names)


if __name__ == "__main__":
    save_dir = 'test'
    main(save_dir=save_dir,
         headless=False)

