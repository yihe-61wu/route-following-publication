import logging
import numpy as np
from simulation_main import *
from video_recorder import Recorder
from plot_from_data import DataVisualiser

from igibson_api import * # to be removed after obstacle integrated

def main(model_name='lamb',
         robot_name='freight',
         scene_name='random',
         save_dir='test',
         headless=True,
         **kwargs):
    ### initialisation
    for key, value in kwargs.items():
        kwargs_main[key] = value
    simulation = SimulationOrganisor(scene_name, robot_name, headless)
    robot = simulation.prepare_robot()
    scene = simulation.prepare_scene(load_object_categories=kwargs_main['load_object_categories'])
    s = simulation.prepare_simulator()
    simulation.initialise_simulation(fix_route=kwargs_main['fix_route'])
    init_pose = simulation.init_robot_pos_orn
    toy_brain = construct_model(model_name, simulation, **kwargs_main)
    var_monitor = ['collision', 'proximity', 'normal', 'v_lin', 'v_ang'] #, 'W_kc2mbon')  # expensive to store W
    recorder = Recorder(simulation, save_dir, *var_monitor)

    #########
    # o_robot = simulation.prepare_dynamical_obstacles(['scaled_freight', 'turtlebot', 'locobot'], [[0, 0, 0], [0, -1, 0], [0.2, -2, 0]])
    # print(o_robot)
    #########

    ### training
    N_train = 1
    print('start training ...')
    for trial in range(N_train):
        print('training trial {}'.format(trial))
        simulation.reinit_robot_pose()
        recorder.start_recording()

        if kwargs_main['fix_route']:
            for t_ctrl in range(simulation.train_time):
                manual_vlva = simulation.manual_action(t_ctrl)
                (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(True, manual_vlva, noisefree=True)

                s.step()
                recorder.recording(v_lin=v_lin, v_ang=v_ang,
                                   collision=collided, proximity=proximal, normal=normal)
        else:
            for pos, orn in zip(*simulation.manual_po):
                perception = toy_brain.learn_pos_orn(pos, orn, noisefree=True, learn=trial==0)

                s.step()
                recorder.recording(v_lin=0, v_ang=0,
                                   collision=0, proximity=0, normal=0)

        # for pos, orn in zip(*simulation.manual_po):
        #     toy_brain.learn_pos_orn(pos, orn)
        #
        #     s.step()
        #     recorder.recording(v_lin=0, v_ang=0,
        #                        collision=0, proximity=0, normal=0)

        # print('final pos', robot.get_position())

        robot.keep_still()
        take_idx = 'train_{}'.format(trial)
        recorder.stop_recording(take_idx)


    ### test
    N_test = 1
    print('start test ...')
    for trial in range(N_test):
        print('test trial {}'.format(trial))
        simulation.reinit_robot_pose(kwargs_main['rand_test_init_orn'], add_moving_obstacle=True)
        # toy_brain.model.reset()   ## reset clock or origin
        recorder.start_recording()

        for _ in range(250):
            (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(False)
            #####
            # o_robot.apply_action([-v_lin, -v_ang])
            #####

            s.step()
            recorder.recording(v_lin=v_lin, v_ang=v_ang,
                               collision=collided, proximity=proximal, normal=normal)

        robot.keep_still()
        take_idx = 'test_{}'.format(trial)
        recorder.stop_recording(take_idx)

    s.disconnect()

    var_names = ['vl', 'va']
    # var_names = []
    painter = DataVisualiser(recorder.save_dir, headless)
    painter.draw_summary(var_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save_dir = 'test'
    for _ in range(1):
        main(model_name='lambon3v',
             robot_name='scaled_freight',
             headless=False,
             rot_coeff=61,
             vl_ctrl='kinetic',
             v_linear=1,
             preprocess='d',
             scene_name='random',     #'Rs_int',  #'Ihlen_0_int', #
             fix_route=False)
