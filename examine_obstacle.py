import logging
import numpy as np
from simulation_main import *
from video_recorder import Recorder
from plot_from_data import DataVisualiser


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
    var_monitor = ['collision', 'force', 'proximity', 'normal']
    recorder = Recorder(simulation, save_dir, *var_monitor)


    ### training
    N_train = 1
    print('start training ...')
    for trial in range(N_train):
        print('training trial {}'.format(trial))
        simulation.reinit_robot_pose()
        recorder.start_recording()

        for t_ctrl in range(500):
            force = toy_brain.sensor_obstacle.feel_force()
            if force >= 10:
                manual_vlva = 0, 1
            else:
                manual_vlva = 1, 0
            (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(False, manual_vlva, noisefree=True)


            s.step()
            recorder.recording(force=force, collision=collided, proximity=proximal, normal=normal)


        robot.keep_still()
        take_idx = 'train_{}'.format(trial)
        recorder.stop_recording(take_idx)


    s.disconnect()

    var_names = var_monitor
    # var_names = []
    painter = DataVisualiser(recorder.save_dir, headless)
    painter.draw_summary(var_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    save_dir = 'test'
    for _ in range(1):
        main(model_name='lamb',
             robot_name='scaled_freight',   #'freight',  #
             headless=False,
             rot_coeff=10,
             v_linear=1,
             scene_name='Pomaria_1_int',    #'stadium', #
             depth_threshold=5,
             obstacle_avoidance=True,
             # load_object_categories='with_no_obj',
             fix_route=True)
