from route_straight_following3in1 import *

repeat = 100

experiment_name = 'straight_route_following'
var_name = 'va_bias'
val_list = 0.6, 0.7, 0.8, 0.9

for _ in range(repeat):
    for val in val_list:
        print('{}={}, trial {}'.format(var_name, val, _))
        save_dir = '{}/{}{}'.format(experiment_name, var_name, val)
        main(save_dir=save_dir,
             headless=True,
             v_linear=0.5,
             noise_motor=0.01,
             va_bias=val,
             scene_name='Ihlen_0_int',
             fix_route=True)


# from route_following_3in1 import *
#
# repeat = 100
#
# experiment_name = 'odometry_noisefreetrain_robustness_2in1_fixedroute'
# var_name = 'noise'
# val_list = 10, 20, 50, 0.1, 0.2, 0.5, 1, 2, 5, 0
#
# for _ in range(repeat):
#     for val in val_list:
#         print('{}={}, trial {}'.format(var_name, val, _))
#         save_dir = '{}/{}{}'.format(experiment_name, var_name, val)
#         main(headless=True,
#              save_dir=save_dir,
#              scene_name='Rs_int',
#              fix_route=True,
#              noise_odometry=val)



# from route_following import *
#
# repeat = 100
#
# experiment_name = 'gridsearch-routefollow/umbkli'
# var_name = 'rot_coeff'
# val_list = 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50
#
# for _ in range(repeat):
#     for switch_step in (5, 10):
#         for val in val_list:
#             print('{}={}, trial {}'.format(var_name, val, _))
#             save_dir = '{}/switch_step{}/{}{}'.format(experiment_name, switch_step, var_name, val)
#             main(model_name='umbkli',
#                  save_dir=save_dir,
#                  headless=True,
#                  scene_name='Rs_int',
#                  obstacle_avoidance=False,
#                  fix_route=True,
#                  rand_test_init_orn=False,
#                  rot_coeff=val,
#                  switch_step=switch_step)