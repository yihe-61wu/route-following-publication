from route_following_multimbon import *

repeat = 100

n_kc = 30000

experiment_name = 'multimbon_random_route_following_{}kc_500epoch'.format(n_kc)

for _ in range(repeat):
    print('trial {}'.format(_))
    save_dir = '{}'.format(experiment_name)
    main(save_dir=save_dir,
         headless=True,
         rot_coeff=5,
         v_linear=1,
         noise_motor=0.01,
         scene_name='random',
         robot_name='scaled_freight',
         fix_route=False,
         N_test=3,
         N_kc=n_kc)