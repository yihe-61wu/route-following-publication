import numpy as np
import matplotlib.pyplot as plt
import os

path_lamb = './records/vary_crot'
path_klino = './records/acain/vary_crot/klino'
path_lambnn = './records/acain/vary_crot/lambnn'

crot = 10

for idx, p in enumerate((path_lamb, path_lambnn, path_klino)):
    path2crot = os.path.join(p, 'crot{}'.format(crot))

    data = np.load(os.path.join(path2crot, os.listdir(path2crot)[0], 'Freight_test_0_record.npz'), allow_pickle=True)
    va = data['extra'][()]['v_lin']
    va_true = data['extra'][()]['v_lin_true']

    plt.figure(idx)
    plt.plot(va)
    plt.plot(va_true)

plt.show()



#### not very useful