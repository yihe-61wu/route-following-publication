import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from mushroom_body import SingleOutputMB
from image_preprocessing import apply_lateral_inhibition
from neuro_utils import random_projection_hash



def show_image(ax, img):
    if img.ndim == 3: img = img[:, :, [2, 1, 0]]
    ax.imshow(img)
    return ax


def transform_image(img, tranform, **kwargs):
    if tranform == 'i':
        im = apply_lateral_inhibition(img, *kwargs['sigma'])
    elif tranform == 'g':
        im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif tranform == 'd':
        im = cv2.resize(img, kwargs['size'])
    elif tranform == 'h':
        # fakeMB = SingleOutputMB(np.prod(kwargs['size_raw']), np.prod(kwargs['size']))
        prelsh = random_projection_hash(np.prod(kwargs['size_raw']), np.prod(kwargs['size']))
        if img.ndim == 2:
            im = prelsh.hashing(img.flatten())
        elif img.ndim == 3:
            im = np.array([prelsh.hashing(img[:, :, i].flatten()) for i in range(3)])
        im = np.multiply(255, im)
    return im


def hist_with_metrics(ax, img):
    n_bin = 64
    pmf, _, _ = ax.hist(img.flatten(), density=True, bins=n_bin)
    flatness = 1 - np.sum((pmf - 1 / n_bin) ** 2) / n_bin
    entropy = -np.sum(pmf * np.log2(pmf + np.finfo(pmf.dtype).eps))
    return ax, flatness, entropy


def prepare_8images(image_raw):
    imgs = {'raw': image_raw}
    for k1 in 'igd':
        imgs[k1] = transform_image(image_raw, k1, sigma=sigma12, size=hw_pn, size_raw=hw_raw)
        for k2 in 'igd':
            if k2 == k1: continue
            imgs[k1 + k2] = transform_image(imgs[k1], k2, sigma=sigma12, size=hw_pn, size_raw=hw_raw)
            for k3 in 'igd':
                if k3 == k1 or k3 == k2: continue
                imgs[k1 + k2 + k3] = transform_image(imgs[k1 + k2], k3, sigma=sigma12, size=hw_pn, size_raw=hw_raw)

    for kk in ('igh', 'gih', 'gh'):
        img = image_raw
        for k0 in kk:
            img = transform_image(img, k0, sigma=sigma12, size=hw_pn, size_raw=hw_raw)
        imgs[kk] = img.reshape(hw_pn)

    return imgs

if __name__ == "__main__":

    # Load an example image
    data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/sim_20240425_184948_Rs_int/tmp_robot_Freight/'
    file_name = '00002.png'
    # file_name = None

    hw_pn = (33, 33)
    sigma12 = 1, 1.6
    # sigma12 = 5, 8

    n_kc = 10000
    mb = SingleOutputMB(np.prod(hw_pn), n_kc, N_pn_perkc=70, sparsity_kc=0.01)
    mbs = SingleOutputMB(33 * 22, n_kc, N_pn_perkc=70, sparsity_kc=0.01)

    if file_name is None:
        final_keys = 'igd', 'idg', 'gid', 'gdi', 'dig', 'dgi', 'igh', 'gih', 'gh', 'gd', 'dg', 'raw'
        nk = len(final_keys)

        data_time = {}
        for k in final_keys:
            data_time[k] = []
        for fn in np.sort(os.listdir(data_dir))[:30]:
            image_raw = cv2.imread(data_dir + fn)
            hw_raw = image_raw.shape[:2]
            imgs = prepare_8images(image_raw)
            for k in final_keys:
                data_time[k].append(imgs[k])

        fig, ax = plt.subplots(7, nk)
        for k, a in zip(final_keys, ax.T):
            if k == 'raw':
                img_hw = image_raw.shape
            else:
                img_hw = hw_pn
            n_p = np.prod(img_hw)
            pn = np.reshape(data_time[k], (-1, n_p)) / 255
            pn_td0 = pn - pn[0]
            if k in ('igh', 'gih', 'gh'):
                dpn = np.count_nonzero(pn_td0, axis=1) / n_p
            else:
                dpn = np.linalg.norm(pn_td0, axis=1) ** 2 / n_p
            a[4].plot(dpn, label='pn')

            avg = np.mean(pn, axis=0)
            std = np.std(pn, axis=0)
            a[1].hist(avg, bins=64, range=(0, 1), density=True)
            a[3].hist(std, bins=64, range=(0, 1), density=True)
            show_image(a[0], avg.reshape(img_hw))
            show_image(a[2], std.reshape(img_hw))
            a[0].set_title(k)
            a[0].set_xticks([])
            a[2].set_xticks([])
            a[0].set_yticks([])
            a[2].set_yticks([])

            if k != 'raw':
                for LI in (False, True):
                    kc = np.array([mb.hashing(pn_t, LI=LI) for pn_t in pn])
                    kc_td0= kc - kc[0]
                    dkc = np.count_nonzero(kc_td0, axis=1) / mb.N_kc_WTA / 2
                    kc_avg = np.mean(kc, axis=0)
                    kc_std = np.std(kc, axis=0)

                    a[4].plot(dkc, label='kc {}'.format(LI))
                    a[5].hist(kc_avg, bins=64, range=(0, 1), density=True)
                    a[6].hist(kc_std, bins=64, range=(0, 1), density=True)
            a[4].legend()

        for ri in (1, 3, 4):
            [a.sharey(ax[ri, 0]) for a in ax[ri]]

    else:

        final_keys = 'igd', 'idg', 'gid', 'gdi', 'dig', 'dgi', 'igh', 'gih', 'gh', 'gd', 'dg'
        nk = len(final_keys)

        image_raw = cv2.imread(data_dir + file_name)
        hw_raw = image_raw.shape[:2]

        # ls_sigma = np.arange(4, 41, 1)

        f1, ax = plt.subplot_mosaic('''
                                    aaabbcdD
                                    aaabbefF
                                    aaagghiI
                                    AA.ggjkK
                                    zzwllmnN
                                    zzyllopP
                                    ''')
        f2, ax2 = plt.subplots(5, 6)

        imgs = prepare_8images(image_raw)

        for ak, (t, im) in zip('abcdefghijklmnopwy', imgs.items()):
            a = show_image(ax[ak], im)
            a.set_title(t)
            a.set_xticks([])
            a.set_yticks([])
            if ak in 'adfiknp':
                b, flat, entropy = hist_with_metrics(ax[ak.upper()], im)
                b.set_xlim(0, 255)
                b.set_ylim(0, 0.02)
                b.text(.5, .6,
                       '1 - flatness {:.2e}\nentropy {:.2f}'.format(1 - flat, entropy),
                       horizontalalignment='center',
                       transform=b.transAxes)

        ls_pn = 'igd', 'idg', 'gid', 'gdi', 'dig', 'dgi'
        for acol, key in zip(ax2.T, ls_pn):
            im = imgs[key]
            a2 = show_image(acol[0], im)
            a2.set_title(key)
            b2, _, _ = hist_with_metrics(acol[1], im)
            b2.set_xlim(0, 255)
            b2.set_ylim(0, 0.02)

            kc = mb.hashing(im.flatten()).reshape(100, 100)
            c2 = show_image(acol[2], kc)
            kc_LI = mb.hashing(im.flatten(), LI=True)
            d2 = show_image(acol[3], kc_LI.reshape(100, 100))
            e2, _, _ = hist_with_metrics(acol[4], kc_LI - kc.flatten())
            e2.set_yscale('log')

        RMSEs = pdist(np.reshape([imgs[k] for k in final_keys], (nk, -1))) ** 2 / np.prod(hw_pn)
        # print(RMSEs)
        ax['z'].imshow(squareform(np.log(RMSEs)))
        ax['z'].set_xticks(np.arange(nk), final_keys)
        ax['z'].set_yticks(np.arange(nk), final_keys)


        ### left right
        ls_pn = 'igd', 'idg', 'gid', 'gdi', 'dig', 'dgi', 'dg', 'gd'
        f3, ax3 = plt.subplots(2, len(ls_pn) + 2)
        for acol, key in zip(ax3.T, ls_pn):
            im = imgs[key]
            a3 = show_image(acol[0], im)
            a3.set_title(key)
            im1eye = [im[:, pxl : pxl + 22].flatten() for pxl in range(33 - 22)]
            iml = np.mean(im1eye[:6], axis=0)
            imr = np.mean(im1eye[6:], axis=0)
            rmse1eye = pdist([iml] + im1eye + [imr]) ** 2 / (33 * 22)
            rmse1eye_sq = squareform(rmse1eye)
            acol[1].imshow(rmse1eye_sq)

            ax3[0, -1].plot(rmse1eye_sq[0] / rmse1eye_sq[0].max(), label=key)
            ax3[1, -1].plot(rmse1eye_sq[-1] / rmse1eye_sq[-1].max(), label=key)

            ax3[0, -2].plot(rmse1eye_sq[1] / rmse1eye_sq[1].max(), label=key)
            ax3[1, -2].plot(rmse1eye_sq[-2] / rmse1eye_sq[-2].max(), label=key)

        ax3[1, -1].legend()

    plt.show()