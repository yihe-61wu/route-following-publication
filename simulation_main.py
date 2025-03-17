from robot_main import *
from igibson_api import NoisyDDController, SimulationOrganisor
from bug_MBCX import *


kwargs_main = {'N_kc': 32000,
               'N_pn_kc': 10,
               'sparsity': 0.01,
               'noise_odometry': 0,
               'noise_vision': 0,
               'noise_motor': 0,
               'v_linear': 0.5,
               'lookaround': 0.5,
               'update_timestep': 1,
               'obstacle_avoidance': True,
               'load_object_categories': None,
               'texture_randomization': False,
               'N_train': 1,
               'N_test': 1,
               'fix_route': False,
               'rand_test_init_orn': False,
               'collision_force_threshold': 10,    # 100 was used; quite constant wrt to a varying robot scale
               'depth_threshold': 1,    # the actual threshold scales with robot scale
               'rot_coeff': None,  ## optimal for lamb in route flowing is 10
               'switch_step': 5,
               'vl_bias': 0.0,
               'va_bias': 0.0,
               'MBONnet': 'random',
               'N_mbon': 4,
               'goal': [0.65678943, -3.14161031], # for Rs_int
               'period': 10, # not optimal
               'range': 0.1, # not optimal
               'lateral_inhibition': None,
               'init_pose': None,
               'preprocess': 'id',
               'sigma12': (1, 2),
               'mindist_start2goal': 3,
               'fix_pos_goal': None
               }


def construct_model(model_name, simulation:SimulationOrganisor, **kwargs):
    simulator, robot = simulation.simulator, simulation.robot
    sensor_vision = LateralisedVisualSensor(simulator, robot,
                                            noise=kwargs['noise_vision'],
                                            lateral_inhibition=kwargs['lateral_inhibition'],
                                            preprocess=kwargs['preprocess'],
                                            sigma12=kwargs['sigma12'])
    sensor_odometry = OracleBasedOdometrySensor(simulator, robot,
                                                kwargs['update_timestep'],
                                                kwargs['noise_odometry'])
    pc_sensor = ProximityCollisionSensor(simulator, robot,
                                         force_threshold=kwargs['collision_force_threshold'],
                                         depth_threshold=kwargs['depth_threshold'])
    ddcontroller = NoisyDDController(robot, noise=kwargs['noise_motor'])
    if model_name == 'pure-pursuit':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 1
        sensor = OracleBasedOdometrySensor(simulator, robot, kwargs['update_timestep'], kwargs['noise_odometry'])
        controller = PurePursuitSteering(robot, kwargs['v_linear'], kwargs['lookaround'], rot_coeff=kwargs['rot_coeff'])
        brain = PurePursuit(sensor, controller)
    elif model_name == 'visual-localisor':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 1
        sensor_vision = ConstrainedVisualSensor(simulator, robot, noise=kwargs['noise_vision'])
        sensor_odometry = OracleBasedOdometrySensor(simulator, robot, kwargs['update_timestep'], kwargs['noise_odometry'])
        controller = PurePursuitSteering(robot, kwargs['v_linear'], kwargs['lookaround'], rot_coeff=kwargs['rot_coeff'])
        brain = NaiveVisualLocalisor(sensor_vision, sensor_odometry, controller)
    elif model_name == 'lamb':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 10
        # MultiOutputMB can reproduce results of SingleOutputMB (not perfect memory)
        # mb_l = SingleOutputMB(sensor_vision.eye_size, kwargs['N_kc'], N_pn_perkc=kwargs['N_pn_kc'], sparsity_kc=kwargs['sparsity'])
        # mb_r = SingleOutputMB(sensor_vision.eye_size, kwargs['N_kc'], N_pn_perkc=kwargs['N_pn_kc'], sparsity_kc=kwargs['sparsity'])
        mb_l = MultiOutputMB(sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        mb_r = MultiOutputMB(sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralFamiliaritySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'])
        # brain = LaMB(mb_l, mb_r, sensor_vision, controller)
        sensor_odometry = OracleBasedOdometrySensor(simulator, robot, kwargs['update_timestep'], kwargs['noise_odometry'])
        brain = LaMBmo(mb_l, mb_r, sensor_vision, controller, odometry=sensor_odometry,
                       MBONnet=kwargs['MBONnet'], goal=kwargs['goal'], period=kwargs['period'], range=kwargs['range'])
        # var_monitor.append(perception)
    elif model_name == 'lambon':
        mb = MultiOutputMB(sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralFamiliaritySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'])
        brain = LaMBON(mb, sensor_vision, controller)
    elif model_name == 'lambon3v':
        mb = MultiOutputMB(sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = Lateral3viewFamiliaritySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'])
        brain = LaMBON3view(mb, sensor_vision, controller)
    elif model_name == 'space-lamb':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 1
        # MultiOutputMB can reproduce results of SingleOutputMB (not perfect memory)
        # mb_l = SingleOutputMB(sensor_vision.eye_size, kwargs['N_kc'], N_pn_perkc=kwargs['N_pn_kc'], sparsity_kc=kwargs['sparsity'])
        # mb_r = SingleOutputMB(sensor_vision.eye_size, kwargs['N_kc'], N_pn_perkc=kwargs['N_pn_kc'], sparsity_kc=kwargs['sparsity'])
        mb_l = MultiOutputMB(sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        mb_r = MultiOutputMB(sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralFamiliaritySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'])
        # brain = LaMB(mb_l, mb_r, sensor_vision, controller)
        sensor_odometry = OracleBasedOdometrySensor(simulator, robot, kwargs['update_timestep'], kwargs['noise_odometry'])
        brain = SpatialLaMB(mb_l, mb_r, sensor_vision, controller, odometry=sensor_odometry)
    elif model_name == 'dimb-add':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 61
        mb_l = MOMBnovelty('superposition', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        mb_r = MOMBnovelty('superposition', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralNoveltySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'], vl_ctrl=kwargs['vl_ctrl'])
        brain = DiMB(mb_l, mb_r, sensor_vision, controller)
    elif model_name == 'dimb-DA':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 10
        mb_l = MOMBnovelty('dopaminergic', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        mb_r = MOMBnovelty('dopaminergic', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralNoveltySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'], vl_ctrl=kwargs['vl_ctrl'])
        brain = DiMB(mb_l, mb_r, sensor_vision, controller)
    elif model_name == 'dimb-RP':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 8
        mb_l = MOMBnovelty('reward-predication', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        mb_r = MOMBnovelty('reward-predication', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralNoveltySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'], vl_ctrl=kwargs['vl_ctrl'])
        brain = DiMB(mb_l, mb_r, sensor_vision, controller)
    elif model_name == 'dimb-ST':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 8
        mb_l = MOMBnovelty('sparse-target', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        mb_r = MOMBnovelty('sparse-target', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralNoveltySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'], vl_ctrl=kwargs['vl_ctrl'])
        brain = DiMB(mb_l, mb_r, sensor_vision, controller, odometry=sensor_odometry, init_pose=simulation.init_robot_pos_orn)
    elif model_name == 'dimb-AN':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 8
        mb_l = MOMBnovelty('all-or-none', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        mb_r = MOMBnovelty('all-or-none', sensor_vision.eye_size, kwargs['N_pn_kc'], kwargs['N_kc'], sparsity_kc=kwargs['sparsity'], N_mbon=kwargs['N_mbon'])
        controller = LateralNoveltySteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'], vl_ctrl=kwargs['vl_ctrl'])
        brain = DiMB(mb_l, mb_r, sensor_vision, controller)
    elif model_name == 'umbkli':
        if kwargs['rot_coeff'] is None: kwargs['rot_coeff'] = 0.3
        sensor_vision = ConstrainedVisualSensor(simulator, robot, noise=kwargs['noise_vision'], preprocess=kwargs['preprocess'], sigma12=kwargs['sigma12'])
        mb = SingleOutputMB(sensor_vision.view_size, kwargs['N_kc'], N_pn_perkc=kwargs['N_pn_kc'], sparsity_kc=kwargs['sparsity'])
        controller = KlinokinesisSteering(kwargs['v_linear'], rot_coeff=kwargs['rot_coeff'], switch_step=kwargs['switch_step'])
        brain = UMBKli(mb, sensor_vision, controller)
    else:
        raise NotImplementedError('Wrong model name!')
    model = EmbodiedSensoryMotorModel(brain, ddcontroller, pc_sensor, kwargs['obstacle_avoidance'])
    return model


def construct_MBCX_model(model_name, simulation:SimulationOrganisor, **kwargs):
    simulator, robot = simulation.simulator, simulation.robot
    sensor_odometry = OracleBasedOdometrySensor(simulator, robot, noise=kwargs_main['noise_odometry'])
    obstacle_sensor = ProximityCollisionSensor(simulator, robot)
    if model_name == 'GoalFollower':
        mbcx_model = Goal_follower(sensor_odometry,
                                   obstacle_sensor,
                                   simulation.pos_goal[:2],
                                   steer_bias=[kwargs['vl_bias'], kwargs['va_bias']])
    else:
        if model_name == 'MB4ONaxial':
            mb_model = MB4ONaxial(simulator, robot,
                                  preprocess=kwargs['preprocess'])
        elif model_name == 'MB2ONopponent':
            mb_model = MB4ONaxial(simulator, robot, lesion=[2, 3],
                                  preprocess=kwargs['preprocess'])
        elif model_name == 'MB2ONbilateral':
            mb_model = MB4ONaxial(simulator, robot, lesion=[0, 1],
                                  preprocess=kwargs['preprocess'])
        elif model_name == 'MB4ONdiagonal':
            mb_model = MB4ONdiagonal(simulator, robot,
                                  preprocess=kwargs['preprocess'])
        elif model_name == 'CXonly':
            mb_model = MB4ONaxial(simulator, robot, lesion=[0, 1, 2, 3],
                                  N_kc=10, N_pn_perkc=10, S_kc=0.1,
                                  preprocess=kwargs['preprocess'])
        mbcx_model = MBCX_driver(sensor_odometry,
                                 obstacle_sensor,
                                 mb_model,
                                 simulation.pos_goal[:2],
                                 steer_bias=[kwargs['vl_bias'], kwargs['va_bias']])
    return mbcx_model