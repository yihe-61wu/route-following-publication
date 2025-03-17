import matplotlib.pyplot as plt
import numpy as np
from simulation_main import *
from video_recorder import Recorder
from plot_from_data import DataVisualiser
from igibson.utils.transform_utils import quat2axisangle, axisangle2quat


def main(robot_name='freight',
         scene_name='Rs_int',
         save_dir='test',
         headless=True,
         **kwargs):
    ### initialisation
    for key, value in kwargs.items():
        kwargs_main[key] = value
    simulation = SimulationOrganisor(scene_name, robot_name, headless)
    robot = simulation.prepare_robot()
    scene = simulation.prepare_scene(load_object_categories=kwargs_main['load_object_categories'],
                                     texture_randomization=kwargs_main['texture_randomization'])
    s = simulation.prepare_simulator()
    simulation.initialise_simulation(fix_route=kwargs_main['fix_route'])
    init_pose = simulation.init_robot_pos_orn
    kwargs_main['init_pose'] = init_pose
    model_list = 'space-lamb', # 'dimb-ST', #'dimb-DA', #'dimb-AN', 'dimb-RP', 'dimb-add', #, 'pure-pursuit', 'visual-localisor'
    toy_brains = [construct_model(model_name, simulation, **kwargs_main) for model_name in model_list]
    var_monitor = ['collision', 'proximity', 'normal', 'v_lin', 'v_ang', 'mb_l', 'mb_r']
    recorder = Recorder(simulation, save_dir, *var_monitor)

    for model in toy_brains:
        print(model.model.MB['l'])
        print(model.model.MB['l'].evaluating)

    ### training
    N_train = 1
    print('start training ...')
    for trial in range(N_train):
        print('training trial {}'.format(trial))
        simulation.reinit_robot_pose()
        for toy_brain in toy_brains:
            toy_brain.last_route_pos = None
        recorder.start_recording()

        if kwargs_main['fix_route']:
            for t_ctrl in range(125):
                if trial < 2:
                    manual_vlva = simulation.manual_action(t_ctrl)
                else:
                    manual_vlva = 0.5, 0.5
                mb_l, mb_r = [], []
                for toy_brain in toy_brains:
                    (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(trial==0, manual_vlva, noisefree=True)
                    mb_l.append(perception[0])
                    mb_r.append(perception[1])

                s.step()
                recorder.recording(mb_l=mb_l, mb_r=mb_r,
                                   v_lin=v_lin, v_ang=v_ang,
                                   collision=collided, proximity=proximal, normal=normal)
        else:
            for pos, orn in zip(*simulation.manual_po):
                mb_l, mb_r = [], []
                for toy_brain in toy_brains:
                    perception = toy_brain.learn_pos_orn(pos, orn, noisefree=True, learn=trial==0)
                    mb_l.append(perception[0])
                    mb_r.append(perception[1])

                s.step()
                recorder.recording(v_lin=0, v_ang=0,
                                   mb_l=mb_l, mb_r=mb_r,
                                   collision=0, proximity=0, normal=0)

        robot.keep_still()
        take_idx = 'train_{}'.format(trial)
        recorder.stop_recording(take_idx)

    # for toy_brain in toy_brains:
    #     w_profile = toy_brain.model.MB['l'].W_kc2mbon.flatten()
    #     plt.hist(np.where(0 < w_profile, np.log(w_profile), 1),
    #              bins=100,
    #              # range=(-10, 1),
    #              histtype='step')
    # plt.show()


    ### test
    N_test = 1
    print('start test ...')
    for trial in range(N_test):
        for toy_brain, model_name in zip(toy_brains, model_list):
            print('test trial {} of {}'.format(trial, model_name))
            simulation.reinit_robot_pose(kwargs_main['rand_test_init_orn'])
            recorder.start_recording()
            mb_l, mb_r = [], []
            for _ in range(500):
                (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(False)
                mb_l, mb_r = perception

                s.step()
                recorder.recording(mb_l=mb_l, mb_r=mb_r,
                                   v_lin=v_lin, v_ang=v_ang,
                                   collision=collided, proximity=proximal, normal=normal)

            robot.keep_still()
            take_idx = 'test_{}_{}'.format(trial, model_name)
            recorder.stop_recording(take_idx)


    ### post-learning analysis
    pts_analysis = (init_pose[0], # [y, -x, 0]_inplot
                    # [0, 0, 0],
                    # [0, -0.5, 0],
                    # [0.5, -1, 0],
                    # [0.6, -1.8, 0],
                    # [0.4, -1.8, 0],
                    # [0.5, -2.5, 0],
                    # [0.8, -3, 0],
                    [0.8, 3, 0]
                    )
    pts_analysis = []
    print('start scanning ...')
    for trial, pos_anls in enumerate(pts_analysis):
        for toy_brain, model_name in zip(toy_brains, model_list):
            print('scanning trial {} of {}'.format(trial, model_name))
            robot.reset()
            robot.set_position(pos_anls)

            for _ in range(20):  # keep still for 2 seconds
                s.step()

            recorder.start_recording()
            for z_ang in range(360):
                (v_lin, v_ang), _, perception, collided, proximal, normal = toy_brain.control(False)
                mb_l.append(perception[0])
                mb_r.append(perception[1])
                robot.keep_still()
                robot.set_orientation(axisangle2quat([0, 0, z_ang / 180 * np.pi]))
                s.step()
                recorder.recording(mb_l=mb_l, mb_r=mb_r,
                                   v_lin=v_lin, v_ang=v_ang,
                                   collision=collided, proximity=proximal, normal=normal)

            robot.keep_still()
            take_idx = 'analysis_{}_{}'.format(trial, model_name)
            recorder.stop_recording(take_idx)

    s.disconnect()

    var_names = []
    painter = DataVisualiser(recorder.save_dir, headless)
    painter.draw_summary(var_names)


if __name__ == "__main__":
    save_dir = 'test3'
    for _ in range(10):
        main(headless=True,
             save_dir=save_dir,
             vl_ctrl='kinetic',   #'differential', # need parameter tuning
             N_mbon=1,
             scene_name='random',
             robot_name='scaled_freight',
             # fix_route=True,
             v_linear=1,
             rot_coeff=1,
             obstacle_avoidance=True,
             N_pn_kc=70,
             sparsity=0.01,
             # N_pn_kc=5000, # 10-20% of kc according to Flo
             preprocess='id',
             sigma12=(1, 2),
             noise_motor=0.0)
