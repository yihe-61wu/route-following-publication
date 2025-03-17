import numpy as np
from scipy.stats import logistic
# from image_preprocessing import apply_lateral_inhibition
from neuro_utils import random_projection_hash, k_winners_take_all_hash, bloom_filter


class SingleOutputMB:
    def __init__(self, N_pn, N_kc, N_pn_perkc=None, sparsity_kc=0.1, N_memory=None, output_familiarity=True):
        self._nothash = N_kc is None
        self._flyhash = not self._nothash and isinstance(N_pn_perkc, int)
        self._oldhash = not self._nothash and not self._flyhash
        self._plastic = N_memory is None
        self._out_fam = output_familiarity

        self.N_pn = N_pn
        self.N_kc = N_pn if self._nothash else N_kc

        if self._nothash:
            self.hash_met = 'raw-data'
            self.S_kc = 1.0
            self.W_pn2kc = np.eye(N_pn)
        elif self._oldhash:
            self.hash_met = 'dense-LSH'
            self.S_kc = 0.5
            self.W_pn2kc = np.random.randn(self.N_pn, self.N_kc)
        elif self._flyhash:
            self.hash_met = 'FlyHash'
            self.S_kc = sparsity_kc
            self.W_pn2kc = (np.random.rand(self.N_pn, self.N_kc) <= (N_pn_perkc / self.N_pn)).astype(np.int8)
            self.N_kc_WTA = int(self.S_kc * self.N_kc)

        if self._plastic:
            self.W_kc2mbon = np.random.randn(self.N_kc) / np.sqrt(self.N_kc)
        else:
            self.N_memo = N_memory
            self.idx_memo = 0
            self.M_kc = np.random.rand(self.N_memo, self.N_kc)
            if self._out_fam:
                self.val_met = lambda x: np.linalg.norm(x) if self._nothash else np.count_nonzero(x)
            else:
                self.val_met = lambda x: np.sqrt(self.N_kc) - np.linalg.norm(x) if self._nothash else self.N_kc - np.count_nonzero(x)
                # not tested for raw images!!!

    def __str__(self):
        type_kc = 'fuzzy' if self._nothash else 'binary'
        N_item = 1 if self._plastic else self.N_memo
        msg = """
        nodes: {} PN => {} {} KC => {} item
        hashing method: {}
        """.format(self.N_pn, self.N_kc, type_kc, N_item, self.hash_met)
        return msg

    def hashing(self, pn, LI=False):
        if self._nothash:
            kc = pn
            # kc = (pn >= 0.5).astype(int)  # not working
        elif self._oldhash:
            kc = (np.dot(pn, self.W_pn2kc) >= 0).astype(np.int8)
        elif self._flyhash:
            z = np.dot(pn, self.W_pn2kc)
            if LI:
                n_kcrow = np.sqrt(z.size)
                if n_kcrow.is_integer():
                    z = apply_lateral_inhibition(z.reshape((int(n_kcrow), int(n_kcrow))), 1, 2).flatten()
                else:
                    raise ValueError('Number of KC is not a perfect square!')
            idx = np.argsort(z)[-self.N_kc_WTA:]
            kc = np.zeros(self.N_kc, dtype=np.int8)
            kc[idx] = 1
        return kc

    def evaluating(self, kc):
        if self._plastic:
            in_mbon = np.dot(kc, self.W_kc2mbon) / self.N_kc
            # valence = logistic.cdf(in_mbon) - 0.5
            valence = np.maximum(in_mbon, 0)
            # valence = in_mbon
        else:
            val_all = [self.val_met(kc - m) for m in self.M_kc]
            valence = np.max(val_all) if self._out_fam else np.min(val_all)
        return valence

    def learning(self, kc):
        if self._plastic:
            dW = kc
            # dW = kc * 2 - 1 (actually implemented in paper 1, but the effect is not obvious due to self-normalisation)
            self.W_kc2mbon = self.W_kc2mbon + dW if self._out_fam else self.W_kc2mbon - dW
        else:
            self.M_kc[self.idx_memo] = kc
            self.idx_memo = (self.idx_memo + 1) % self.N_memo


class GroundCoordinateMB:
    def __init__(self, N_pn, N_kc, z_pn, N_pn_perkc, S_kc):
        self.N_pn = N_pn
        self.N_kc = N_kc
        self.z_pn = z_pn
        self.S_kc = S_kc
        self.lsh = random_projection_hash(3, N_pn)
        # self.lsh.W[-1] = np.random.rand(N_pn) * z_pn - z_pn / 2  # random bias/ intersection of boundary line
        self.W_pn2kc = (np.random.rand(self.N_pn, self.N_kc) <= (N_pn_perkc / self.N_pn)).astype(np.int8)
        self.N_kc_WTA = int(self.S_kc * self.N_kc)
        self.W_kc2mbon = np.random.randn(self.N_kc) / np.sqrt(self.N_kc)

    def hashing(self, xy):
        xy = np.atleast_2d(xy)
        xyz = np.zeros((xy.shape[0], 3))
        xyz[:, :-1] = xy
        xyz[:, -1] = self.z_pn
        # xyz[:, -1] = np.sqrt(self.z_pn ** 2 - np.sum(xy ** 2, axis=1))
        pn = self.lsh.hashing(xyz)
        kcin = np.matmul(pn, self.W_pn2kc)
        idx = np.argsort(kcin, axis=1)[:, -self.N_kc_WTA:]
        kc = np.zeros((xy.shape[0], self.N_kc))
        # for i in range(xy.shape[0]):
        #     kc[i, idx[i]] = 1
        kc[np.vstack((np.arange(idx.size), idx))] = 1
        return kc

    def evaluating(self, xy):
        kc = self.hashing(xy)
        mbon = np.matmul(kc, self.W_kc2mbon) / self.N_kc_WTA / 2
        return mbon

    def learning(self, xy):
        kc = self.hashing(xy)
        dW = np.mean(np.atleast_2d(kc), axis=0)
        self.W_kc2mbon += dW


class SpatialHashMB:
    def __init__(self, N_pn, N_kc, max_xy, N_pn_perkc, S_kc, learning_decay_rate):
        # self.N_kc = N_kc
        self.N_kc_WTA = int(S_kc * N_kc)
        # self.W_kc2mbon = np.random.randn(self.N_kc) / np.sqrt(self.N_kc)
        # self.W_kc2mbon = np.ones(self.N_kc)
        self.learning_rate = learning_decay_rate

        self.xy2pn = random_projection_hash(2, N_pn, max_xy)
        self.pn2kc = k_winners_take_all_hash(N_pn, N_kc, N_pn_perkc / N_pn, self.N_kc_WTA)
        self.kc2mbon = bloom_filter('decay', N_kc, 1)

    def hashing(self, xy):
        pn = self.xy2pn.hashing(xy)
        kc = self.pn2kc.hashing(pn)
        return kc

    def evaluating(self, xy):
        kc = self.hashing(xy)
        # mbon = np.matmul(kc, self.W_kc2mbon) / self.N_kc_WTA
        mbon = self.kc2mbon.evaluating(kc, self.N_kc_WTA).flatten()
        return mbon

    def learning(self, xy):
        kc = self.hashing(xy)
        # kc_count = np.sum(np.atleast_2d(kc), axis=0)
        # self.W_kc2mbon *= self.learning_rate ** kc_count
        for kk in kc:
            self.kc2mbon.learning(np.atleast_2d(kk), self.learning_rate)


class SHMBensemble:
    def __init__(self, N_ensemble, N_pn, N_kc, max_xy, N_pn_perkc, S_kc, learning_rate):
        self.N_ensemble = N_ensemble
        parameters = [self._init_ensemble_parameters(x) for x in (N_pn, N_kc, max_xy, N_pn_perkc, S_kc, learning_rate)]
        self.MBs = [SpatialHashMB(*x) for x in zip(*parameters)]

    def _init_ensemble_parameters(self, x):
        x = np.ravel(x)
        if len(x) == 1:
            y = np.repeat(x, self.N_ensemble)
        elif len(x) == self.N_ensemble:
            y = np.array(x)
        else:
            raise ValueError('The number of ensembles are not consistent with the number of parameters!')
        return y

    def evaluating(self, xy):
        mbons = [MB.evaluating(xy) for MB in self.MBs]
        return np.mean(mbons, axis=0)

    def learning(self, xy):
        [MB.learning(xy) for MB in self.MBs]

class GroundCoodinateMB22(GroundCoordinateMB):
    def __init__(self, N_pn, N_kc, z_pn, N_pn_perkc, S_kc):
        self.N_pn = N_pn
        self.N_kc = N_kc
        self.z_pn = z_pn
        self.S_kc = S_kc
        self.N_xy = 100
        self.W_xy2pn = np.random.randn(self.N_xy, N_pn)
        self.b_xy = np.random.choice([z_pn, -z_pn], size=self.N_xy)
        # self.b_xy = np.random.rand(self.N_xy) * z_pn - z_pn / 2
        self.W_pn2kc = (np.random.rand(self.N_pn, self.N_kc) <= (N_pn_perkc / self.N_pn)).astype(np.int8)
        self.N_kc_WTA = int(self.S_kc * self.N_kc)
        self.W_kc2mbon = np.random.randn(self.N_kc) / np.sqrt(self.N_kc)

    def hashing(self, xy):
        xy = np.repeat(np.atleast_2d(xy), self.N_xy // 2, axis=1) - self.b_xy
        pnin = np.matmul(xy, self.W_xy2pn)
        pn = (pnin >= 0).astype(int)
        kcin = np.matmul(pn, self.W_pn2kc)
        idx = np.argsort(kcin, axis=1)[:, -self.N_kc_WTA:]
        kc = np.zeros((xy.shape[0], self.N_kc))
        for i in range(xy.shape[0]):
            kc[i, idx[i]] = 1
        return kc


class GroundCoodinateMB3(GroundCoordinateMB):
    def hashing(self, xy):
        r2 = self.z_pn ** 2
        XY = np.atleast_2d(xy)
        XY2 = np.linalg.norm(XY, axis=1) ** 2
        x = 2 * XY[:, 0] * r2 / (XY2 + r2)
        y = 2 * XY[:, 1] * r2 / (XY2 + r2)
        z = self.z_pn * (XY2 - r2) / (XY2 + r2)
        xyz = np.vstack((x, y, z)).T
        pn = self.lsh.hashing(xyz)
        kcin = np.matmul(pn, self.W_pn2kc)
        idx = np.argsort(kcin, axis=1)[:, -self.N_kc_WTA:]
        kc = np.zeros((xy.shape[0], self.N_kc))
        for i in range(xy.shape[0]):
            kc[i, idx[i]] = 1
        return kc



class MultiOutputMB:
    def __init__(self, N_pn, N_pn_perkc, N_kc, sparsity_kc=0.1, W_kc2mbon='rand', N_mbon=1):
        # network size
        self.N_pn, self.N_kc, self.N_mbon = N_pn, N_kc, N_mbon
        # weight matrix
        self.W_pn2kc = (np.random.rand(self.N_pn, self.N_kc) <= (N_pn_perkc / self.N_pn)).astype(np.int8)
        if W_kc2mbon == 'rand':
            self.W_kc2mbon = np.random.randn(self.N_kc, self.N_mbon) / np.sqrt(self.N_kc)
        elif W_kc2mbon == 'zero':
            self.W_kc2mbon = np.zeros((self.N_kc, self.N_mbon))
        else:
            self.W_kc2mbon = W_kc2mbon  # a correct shape is required
        # kc sparisty
        self.S_kc = sparsity_kc
        self.N_kc_WTA = int(self.S_kc * self.N_kc)

    def __str__(self):
        n_pn_pkc, n_kc_ppn = [np.mean(np.count_nonzero(self.W_pn2kc, axis=k)) for k in (0, 1)]
        std_W_kc2mbon = np.std(self.W_kc2mbon)
        msg = """
        nodes: {} PN => {} KC => {} MBON
        on average: 
            {} PN per KC, {} KC per PN
            initial std of W_kc2mbon is {}
        """.format(self.N_pn, self.N_kc, self.N_mbon, n_pn_pkc, n_kc_ppn, std_W_kc2mbon)
        return msg

    def hashing(self, pn, learning=None):
        kcin = np.dot(pn, self.W_pn2kc) # this multiplication should be accelerated as self.W_pn2kc is binary
        idx = np.argsort(kcin)[-self.N_kc_WTA:]
        kc = np.zeros(self.N_kc, dtype=np.int8)
        kc[idx] = 1
        return kc

    def evaluating(self, kc):
        mbonin = np.matmul(kc, self.W_kc2mbon) / self.N_kc # this multiplication should be accelerated as kc is binary
        valence = np.maximum(mbonin, 0)  # used in paper 1
        # valence = mbonin  # parhaps more reasonable for multi-mbon
        return valence

    def learning(self, kc, mbon_rate=None):
        if mbon_rate is None or np.all(mbon_rate)==0: mbon_rate = np.ones(self.N_mbon)
        dW = np.outer(kc, mbon_rate) #* 2 - 1 # for precisely reproducing paper1 using SingleOutputMB
        self.W_kc2mbon += dW


class MOMBnovelty(MultiOutputMB):
    def __init__(self, learning_method,
                 N_pn, N_pn_perkc, N_kc, sparsity_kc=0.1, W_kc2mbon='rand', N_mbon=1,
                 adaptiveKCthreshold=False):
        super().__init__(N_pn, N_pn_perkc, N_kc, sparsity_kc=sparsity_kc, W_kc2mbon='rand', N_mbon=1)
        self.N_pn_perkc = N_pn_perkc
        self.W_kc2mbon = np.ones((self.N_kc, self.N_mbon))
        self.b_mbon = np.zeros(self.N_mbon)
        if adaptiveKCthreshold:
            self.b_kc = np.full(self.N_kc, 0.0)#N_pn_perkc / 2)
            rate_adapt = 0.001
            self.increase = rate_adapt
            self.decrease = rate_adapt * (1 - 1 / (1 - sparsity_kc))
            self.hashing = self._adahash
        else:
            self.b_kc = np.zeros(self.N_kc)
        self.lm = learning_method
        if learning_method == 'superposition':
            # self.learning = super().learning
            self.evaluating = self.superposition_evaluating
        elif learning_method == 'dopaminergic':
            self.learning = self.dopaminergic_learning
            self.evaluating = self._kc2mbon
        elif learning_method == 'reward-predication':
            self.learning = self.rewardpredic_learning
            self.evaluating = self._kc2mbon
        elif learning_method == 'sparse-target':
            self.learning = self.sparsetarget_learning
            self.rW_de = 0.6
            self.rW_in = (1 / self.rW_de) ** (sparsity_kc / (1 - sparsity_kc))
            self.evaluating = self._kc2mbon
        elif learning_method == 'all-or-none':
            self.learning = self.allornone_learning
            self.evaluating = self._kc2mbon


    def _kc2mbon(self, kc):
        if np.sum(kc) == 0:
            mbonin = 0
        else:
            mbonin = np.matmul(kc, self.W_kc2mbon) / self.N_kc_WTA
        return mbonin - self.b_mbon

    def _adahash(self, pn, learning):
        kcin = np.dot(pn, self.W_pn2kc) - self.b_kc
        kc = np.array(kcin).astype(int)
        if learning:
            db = np.multiply(self.increase * kc + self.decrease * (1 - kc), 1 + np.abs(kcin))
            self.b_kc += db
        print(np.max(self.b_kc), np.median(self.b_kc), np.min(self.b_kc), np.sum(kc) / self.N_kc_WTA)
        return kc

    # WTA output, but threshod chenages to match WTA
    # def _adahash(self, pn, learning):
    #     kcin = np.dot(pn, self.W_pn2kc) - self.b_kc
    #     idx_win = np.argsort(kcin)[-self.N_kc_WTA:]
    #     kc = np.zeros(self.N_kc, dtype=np.int8)
    #     kc[idx_win] = 1
    #
    #     if learning: # slightly better
    #         idx_pos = np.array(kcin >= 0).astype(int)
    #         kc_pos = np.zeros(self.N_kc)
    #         kc_pos[idx_pos] = 1
    #         kcdiff = kc - kc_pos
    #         db_kc = np.zeros(self.N_kc)
    #         db_kc[kcdiff == 1] = self.increase
    #         db_kc[kcdiff == -1] = self.decrease
    #         self.b_kc += np.multiply(db_kc, 1 + np.abs(kcin))
    #         print('bias', np.max(self.b_kc), np.median(self.b_kc), np.min(self.b_kc))
    #     return kc

    def superposition_evaluating(self, kc, rate=1):
        mbon = self._kc2mbon(kc) #* self.N_kc_WTA
        mbon = np.maximum(mbon, 0)  # used in paper 1
        novelty = np.exp(-mbon)# * rate)
        return novelty

    def novelty_evaluating(self, kc):
        mbon = self._kc2mbon(kc)
        mbon = np.maximum(mbon, 0)  # used in paper 1
        return mbon

    def dopaminergic_learning(self, kc, mbon, mbon_rate=None):
        if mbon_rate is None: mbon_rate = np.ones(self.N_mbon) #* 1 # to be optimised
        dW = -np.multiply(mbon, mbon_rate) * (np.tile(kc, (self.N_mbon, 1)).T + self.W_kc2mbon - 1)
        # self.W_kc2mbon = np.clip(self.W_kc2mbon + dW, 0, 1)
        self.W_kc2mbon = np.maximum(0, self.W_kc2mbon + dW)

    def rewardpredic_learning(self, kc, mbon, mbon_rate=None):
        if mbon_rate is None: mbon_rate = np.ones(self.N_mbon) #* 0.1 # okay; to be optimised
        dW = -np.multiply(mbon, mbon_rate) * np.tile(kc, (self.N_mbon, 1)).T
        self.W_kc2mbon = np.clip(self.W_kc2mbon + dW, 0, 1)

    def allornone_learning(self, kc, mbon, mbon_rate=None): # identical to above
        if mbon_rate is None: mbon_rate = np.ones(self.N_mbon)
        self.W_kc2mbon -= np.multiply(1, mbon_rate) * np.tile(kc, (self.N_mbon, 1)).T
        self.W_kc2mbon = np.maximum(0, self.W_kc2mbon)

    def sparsetarget_learning(self, kc, mbon, mbon_rate=None):
        if mbon_rate is None: mbon_rate = np.ones(self.N_mbon)
        rW = kc * self.rW_de + (1 - kc) * self.rW_in
        W = np.multiply(self.W_kc2mbon, np.tile(rW, (self.N_mbon, 1)).T)
        self.W_kc2mbon = np.clip(W, 0, 1)


if __name__ == '__main__':
    N_pn, N_kc = 20, 200
    N_pn_perkc = 10
    sparsity_kc = 0.1

    z_pn = 10
    sMB = GroundCoordinateMB(N_pn, N_kc, z_pn, N_pn_perkc, S_kc=sparsity_kc)
    print(sMB.evaluating([[0, 0], [1, 2], [2, -1], [30, 3]]))
    sMB.learning([[2, -1], [3, 3]])
    print(sMB.evaluating([[0, 0], [1, 2], [2, -1], [30, 3]]))

    # sMB = SingleOutputMB(N_pn, N_kc, N_pn_perkc=N_pn_perkc, sparsity_kc=sparsity_kc)
    # lsh = SingleOutputMB(N_pn, N_kc, N_pn_perkc=None)
    #
    # N_xy = 10
    # xy = np.empty((N_xy, 2))
    # xy[:, 1] = 100  # this constant affects result
    # xy[:, 0] = np.arange(N_xy) * 10
    # dist_xy = np.linalg.norm(xy[1:] - xy[0], axis=1)
    # print('d xy', dist_xy)
    #
    # pn = np.tile(xy, (1, N_pn // 2))
    # kc = np.array([sMB.hashing(p) for p in pn])
    # dist_kc = np.count_nonzero(kc[1:] - kc[0], axis=1) / sMB.N_kc_WTA / 2
    # print('d kc', dist_kc)
    #
    # code_lsh = np.array([lsh.hashing(p) for p in pn])
    # dist_lsh = np.count_nonzero(code_lsh[1:] - code_lsh[0], axis=1) / sMB.N_kc
    # print('d lsh', dist_lsh)
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 2)
    # ax[0].plot(dist_kc)
    # ax[1].plot(dist_lsh)
    # plt.show()

    # arctan_pn = np.empty((N_xy, 0))
    # for angle_n in range(N_pn // 2):
    #     arctan_xy = np.arctan(xy / (angle_n + 1))
    #     arctan_pn = np.hstack((arctan_pn, arctan_xy))
    # dist_apn = np.linalg.norm(arctan_pn[1:] - arctan_pn[0], axis=1)
    # print('d apn', dist_apn)
    #
    # akc = np.array([sMB.hashing(p) for p in arctan_pn])
    # dist_akc = np.count_nonzero(akc[1:] - akc[0], axis=1) / sMB.N_kc_WTA / 2
    # print('d akc', dist_akc)


    # mMB = MOMBnovelty('sparse-target', N_pn, N_pn_perkc, N_kc, sparsity_kc=sparsity_kc, N_mbon=1, W_kc2mbon='rand')
    # print(mMB)
    # print(mMB.W_pn2kc)
    # print(np.sum(mMB.W_pn2kc, axis=1))
    #
    #
    # pn = np.linspace(0, 1, 10)
    # kc = mMB.hashing(pn)
    # print('kc', kc)
    # mbon = mMB.evaluating(kc)
    # print(mbon)
    #
    # [mMB.learning(kc, [1, 0.5]) for _ in range(5)]
    # mbon = mMB.evaluating(kc)
    # print(mbon)
