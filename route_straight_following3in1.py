import logging
import numpy as np
from simulation_main import *
from video_recorder import Recorder
from plot_from_data import DataVisualiser


def main(robot_name='freight',
         scene_name='random',
         save_dir='test',
         headless=True,
         **kwargs):
    ### initialisation
    for key, value in kwargs.items():
        kwargs_main[key] = value
    kwargs_main['obstacle_avoidance'] = False
    simulation = SimulationOrganisor(scene_name, robot_name, headless)
    robot = simulation.prepare_robot()
    scene = simulation.prepare_scene(load_object_categories=kwargs_main['load_object_categories'])
    s = simulation.prepare_simulator()
    simulation.initialise_simulation(fix_route=kwargs_main['fix_route'])
    init_pose = simulation.init_robot_pos_orn
    model_list = 'lamb', 'pure-pursuit', 'visual-localisor'
    toy_brains = [construct_model(model_name, simulation, **kwargs_main) for model_name in model_list]
    var_monitor = ['collision', 'proximity', 'normal', 'v_lin', 'v_ang'] #, 'W_kc2mbon')  # expensive to store W
    recorder = Recorder(simulation, save_dir, *var_monitor)


    ### training
    N_train = 1
    print('start training ...')
    for trial in range(N_train):
        print('training trial {}'.format(trial))
        simulation.reinit_robot_pose()
        recorder.start_recording()

        for t_ctrl in range(180):
            manual_vlva = simulation.manual_action(t_ctrl)
            for toy_brain in toy_brains:
                (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(True, manual_vlva, noisefree=True)

            s.step()
            recorder.recording(v_lin=v_lin, v_ang=v_ang,
                               collision=collided, proximity=proximal, normal=normal)


        robot.keep_still()
        take_idx = 'train_{}'.format(trial)
        recorder.stop_recording(take_idx)

    ### test - model off
    N_test = 1
    print('start baseline ...')
    for trial in range(N_test):
        print('baseline trial {}'.format(trial))
        simulation.reinit_robot_pose(kwargs_main['rand_test_init_orn'])
        recorder.start_recording()

        for _ in range(250):
            manual_vlva = simulation.manual_action(t_ctrl)
            (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(False, manual_vlva, va_bias=kwargs_main['va_bias'])

            s.step()
            recorder.recording(v_lin=v_lin, v_ang=v_ang,
                               collision=collided, proximity=proximal, normal=normal)

        robot.keep_still()
        take_idx = 'base_{}'.format(trial)
        recorder.stop_recording(take_idx)

    ### test
    N_test = 1
    print('start test ...')
    for trial in range(N_test):
        for toy_brain, model_name in zip(toy_brains, model_list):
            print('test trial {} of {}'.format(trial, model_name))
            simulation.reinit_robot_pose(kwargs_main['rand_test_init_orn'])
            recorder.start_recording()

            for _ in range(250):
                (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(False, va_bias=kwargs_main['va_bias'])

                s.step()
                recorder.recording(v_lin=v_lin, v_ang=v_ang,
                                   collision=collided, proximity=proximal, normal=normal)

            robot.keep_still()
            take_idx = 'test_{}_{}'.format(trial, model_name)
            recorder.stop_recording(take_idx)

    s.disconnect()

    # var_names = ['vl', 'v_lin', 'va', 'v_ang']
    var_names = []
    painter = DataVisualiser(recorder.save_dir, headless)
    painter.draw_summary(var_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save_dir = 'test'
    for _ in range(1):
        main(headless=False,
             v_linear=0.5,
             noise_motor=0.01,
             va_bias=-0.1,
             scene_name='Ihlen_0_int', #'Rs_int',
             fix_route=True)
