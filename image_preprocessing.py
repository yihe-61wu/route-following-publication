import cv2
import numpy as np


def apply_lateral_inhibition(image, sigma_center, sigma_surround):
    """
    Apply lateral inhibition using the Difference of Gaussians (DoG) model.

    Args:
        image (numpy.ndarray): Input image
        sigma_center (float): Standard deviation of the center (excitatory) Gaussian kernel
        sigma_surround (float): Standard deviation of the surround (inhibitory) Gaussian kernel

    Returns:
        numpy.ndarray: Image with lateral inhibition applied
    """
    # Create Gaussian kernels
    size = int(max(sigma_center, sigma_surround) * 6) + 1
    center_kernel = cv2.getGaussianKernel(size, sigma_center)
    surround_kernel = cv2.getGaussianKernel(size, sigma_surround)

    # Create 2D Gaussian kernels
    center_kernel = center_kernel * center_kernel.T
    surround_kernel = surround_kernel * surround_kernel.T

    # Apply DoG model
    center_response = cv2.filter2D(image, -1, center_kernel)
    surround_response = cv2.filter2D(image, -1, surround_kernel)
    inhibited_image = center_response - surround_response

    return inhibited_image



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.spatial.distance import pdist, squareform
    from mushroom_body import SingleOutputMB

    # Load an example image
    data_dir = '/home/yihelu/miniconda3/envs/igibson/lib/python3.8/site-packages/igibson/yihe/records/test/sim_20240425_184948_Rs_int/tmp_robot_Freight/'
    image_raw = cv2.imread(data_dir + '00102.png')

    hw_pn = (33, 33)
    sigma12 = 1, 2

    fakeMB = SingleOutputMB(np.prod(image_raw.shape[:2]), np.prod(hw_pn))

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
            if img.ndim == 2:
                im = fakeMB.hashing(img.flatten())
            elif img.ndim == 3:
                im = np.array([fakeMB.hashing(img[:, :, i].flatten()) for i in range(3)])
            im = np.multiply(255, im)
        return im

    def hist_with_metrics(ax, img):
        n_bin = 64
        pmf, _, _ = ax.hist(img.flatten(), density=True, bins=n_bin)
        flatness = 1 - np.sum((pmf - 1 / n_bin) ** 2) / n_bin
        entropy = -np.sum(pmf * np.log2(pmf + np.finfo(pmf.dtype).eps))
        return ax, flatness, entropy


    # ls_sigma = np.arange(4, 41, 1)

    f1, ax = plt.subplot_mosaic('''
                                aaabbcdD
                                aaabbefF
                                aaagghiI
                                AA.ggjkK
                                zzwllmnN
                                zzyllopP
                                ''')

    imgs = {'raw': image_raw}
    for k1 in 'igd':
        imgs[k1] = transform_image(image_raw, k1, sigma=sigma12, size=hw_pn)
        for k2 in 'igd':
            if k2 == k1: continue
            imgs[k1 + k2] = transform_image(imgs[k1], k2, sigma=sigma12, size=hw_pn)
            for k3 in 'igd':
                if k3 == k1 or k3 == k2: continue
                imgs[k1 + k2 + k3] = transform_image(imgs[k1 + k2], k3, sigma=sigma12, size=hw_pn)

    for kk, ak in zip(('igh', 'gih'), 'wy'):
        img = image_raw
        for k0 in kk:
            img = transform_image(img, k0, sigma=sigma12, size=hw_pn)
        imgs[kk] = img.reshape(hw_pn)
        a = show_image(ax[ak], imgs[kk])
        a.set_title(kk)
        a.set_xticks([])
        a.set_yticks([])

    for ak, (t, im) in zip('abcdefghijklmnop', imgs.items()):
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


    keys = 'igd', 'idg', 'gid', 'gdi', 'dig', 'dgi', 'igh', 'gih'
    nk = len(keys)
    RMSEs = pdist(np.reshape([imgs[k] for k in keys], (nk, -1))) ** 2 / np.multiply(*hw_pn)
    print(RMSEs)
    ax['z'].imshow(squareform(np.log(RMSEs)))
    ax['z'].set_xticks(np.arange(nk), keys)
    ax['z'].set_yticks(np.arange(nk), keys)


    # fig, axes = plt.subplots(1, 3)
    # # Apply lateral inhibition
    # for sigma_center in (1,):# 2, 3):
    #     for sigma_surround in (1.6, 5, 20, 50):# np.flip(ls_sigma):
    #         inhibited_image = apply_lateral_inhibition(image, sigma_center, sigma_surround)
    #         ht = axes[0].hist(inhibited_image.flatten(), label='{}-{}'.format(sigma_center, sigma_surround),
    #                  bins=64, range=(0,255), histtype='step')
    #         axes[0].axvline(np.mean(inhibited_image), color=ht[-1][0].get_edgecolor())
    # axes[0].hist(image.flatten(), label='raw image',
    #          bins=64, range=(0,255), histtype='step', color='k')
    # axes[0].axvline(np.mean(image), color='k')
    # axes[0].legend()
    # axes[0].set_xticks(np.arange(0, 256, 32))
    #
    # axes[1].imshow(image)
    # axes[1].set_title('Original Image')
    #
    # axes[2].imshow(inhibited_image)
    # axes[2].set_title('Lateral Inhibition (DoG)')
    plt.show()

    # import os
    # fig, ax = plt.subplot_mosaic('''ab
    #                                 ac''')
    # m_raw, m_LI = [], []
    # for file in os.listdir(data_dir):
    #     image = cv2.resize(cv2.imread(os.path.join(data_dir, file), cv2.IMREAD_GRAYSCALE), (33, 33))
    #     m_raw.append(np.std(image))
    #     m_lii = []
    #     for sigma_surround in ls_sigma:
    #         inhibited_image = apply_lateral_inhibition(image, 1, sigma_surround)
    #         m_lii.append((np.std(inhibited_image)))
    #         m_LI.append(m_lii)
    #
    # m_sort = np.argsort(m_raw)
    # mr, ml = np.array(m_raw)[m_sort], np.array(m_LI)[m_sort]
    # ax['a'].plot(mr, label='raw')
    # pixelmean, pixel_std = [], []
    # for yy, sigma_surround in zip(ml.T, ls_sigma):
    #     pixelmean.append(np.mean(yy))
    #     pixel_std.append(np.std(yy))
    #     ax['a'].plot(yy, label='sigma={}, mean={:.2f}'.format(sigma_surround, pixelmean[-1]))
    #
    # ax['b'].plot(ls_sigma, pixelmean, label='avg')
    # ax['c'].plot(ls_sigma, pixel_std, label='std')
    #
    # for a in ax.values():
    #     a.legend()
    #     a.grid()
    #
    # ax['a'].set_xlabel('image ID sorted by pixel std')
    # ax['a'].set_ylabel('pixel std (a.u.)')
    # for k in 'bc':
    #     ax[k].set_xlabel('sigma surround')
    #     ax[k].set_ylabel('a.u.')
    # plt.show()