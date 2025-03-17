import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import fftpack as fft


dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/rotate_data/sim_20230509_124523_Rs_int/tmp_robot_Freight'
def imread(img_idx, lr):
    path = os.path.join(dir, '{:05d}.png'.format(img_idx))
    img_raw = cv2.imread(path)
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    img_resize = cv2.resize(img_gray, (33, 33))
    img_blur = cv2.blur(img_resize, (7, 7))
    if lr == 'l':
        img = img_blur[:, :22]
    elif lr == 'r':
        img = img_blur[:, -22:]
    return img


def pn2kc(pn, W):
    _nkc = W.shape[1]
    kc = np.zeros(_nkc)
    activation = np.dot(pn, W)
    winner_idx = np.argsort(activation)
    kc[winner_idx[-_nkc//10:]] = 1
    return kc

def lsh(X, W):
    activation = np.dot(X, W)
    return (activation > 0).astype(int)

for lr in 'l':
    idx_ref = 0
    img_ref = imread(idx_ref, lr)
    img_ref = (np.floor(np.random.normal(img_ref, 0.1))).astype(int)
    # cv2.imshow('reference image', img_ref)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    n_kc = 5000
    w_pn2kc = (np.random.rand(img_ref.size, n_kc) <= 0.01).astype(int)
    kc_ref = pn2kc(img_ref.flatten(), w_pn2kc)

    # lsh, fourier
    ft_ref = fft.fft(img_ref)
    dct_ref = fft.dct(img_ref)

    hash_len = 500
    w_lsh = np.random.rand(img_ref.size, hash_len) - 0.5
    hash_ref = lsh(img_ref.flatten(), w_lsh)

    dist = []
    d_kc = []
    d_ft = []
    d_dct = []
    d_lsh = []
    for i in range(360):
        img_now = imread(i, lr)
        dist.append(np.linalg.norm(img_now - img_ref) / np.sqrt(img_ref.size))
        d_kc.append(np.count_nonzero(kc_ref == pn2kc(img_now.flatten(), w_pn2kc)) / n_kc)
        d_ft.append(np.linalg.norm(fft.fft(img_now) - ft_ref) / np.sqrt(ft_ref.size))
        d_dct.append(np.linalg.norm(fft.dct(img_now) - dct_ref) / np.sqrt(dct_ref.size))
        d_lsh.append(np.count_nonzero(hash_ref == lsh(img_now.flatten(), w_lsh)) / hash_len)

    kc_dist = 1 - np.array(d_kc)
    lsh_dist = 1 - np.array(d_lsh)
    idx_0 = (idx_ref - 180 + 30) % 360 # +30 for plotting only MB meeting
    nor_l2, nor_kc, nor_ft, nor_dct, nor_lsh = [np.concatenate((dd[idx_0:], dd[:idx_0])) / np.max(dd) for dd in (dist, kc_dist, d_ft, d_dct, lsh_dist)]

    plt.plot(1 - nor_l2, label='L2', marker='o', c='gray', mfc='none')
    plt.plot(1 - nor_ft, label='L2-DFT', c='b')
    plt.plot(1 - nor_dct, label='L2-DCT', c='g')
    plt.plot(1 - nor_lsh, label='Hamming-LSH', c='orange')
    plt.plot(1 - nor_kc, label='Hamming-FlyHash', c='red')

plt.axvline(180, color='k')
plt.xticks(np.arange(0, 361, 60), np.arange(-180, 181, 60))
plt.legend()
plt.grid()
plt.ylabel(r'normalised similarity $= 1 -$ normalised distance')
plt.xlabel('angular offset from correct orientation (deg)')
plt.title('right familiarity (1 view learned)')

# idx_gap = 60
# bias = 0.0
# rot = []
# rot0 = []
# rot2 = []
# for j in range(len(d_kc)):
#     l, r = d_kc[(j - idx_gap // 2) % 360] - bias, d_kc[(j + idx_gap // 2) % 360] - bias
#     rot0.append(r - l)
#     rot.append((r - l) / (r + l))
#     rot2.append((r - l) / np.sqrt((r ** 2 + l ** 2)))
#
# plt.figure('normalised compare')
# plt.plot(rot / np.max(rot), label='r-l/r+l')
# plt.plot(rot0 / np.max(rot0), label='r-l')
# plt.plot(rot2 / np.max(rot2), label='r-l/sqrt(r2+l2)')
# plt.axvline(180, color='k')
# plt.xticks(np.arange(0, 361, 30), np.arange(-180, 181, 30))
# plt.legend()
# plt.grid()
#
plt.show()