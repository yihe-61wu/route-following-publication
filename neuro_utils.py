import numpy as np


class random_projection_hash:
    def __init__(self, dim_data, dim_hash, max_hash_bias=0):
        W_d2h = np.random.randn(dim_data, dim_hash)
        self.W_data2hash = np.divide(W_d2h, np.linalg.norm(W_d2h, axis=0))
        self.b_hash = (np.random.rand(dim_hash) * 2 - 1) * max_hash_bias

    def __str__(self):
        msg = '''
        class: {}
        dim_data = {}, dim_hash = {}, min_max_hash_bias = {}, {}
        '''.format(self.__class__.__name__, *self.W_data2hash.shape, np.min(self.b_hash), np.max(self.b_hash))
        return msg

    def hashing(self, data):
        hash = (np.matmul(np.atleast_2d(data), self.W_data2hash) >= self.b_hash).astype(int)
        return hash


class k_winners_take_all_hash:
    def __init__(self, dim_data, dim_hash, sparsity_W, N_winner):
        self.W_data2hash = (np.random.rand(dim_data, dim_hash) <= sparsity_W).astype(int)
        self.N_winner = N_winner

    def __str__(self):
        msg = '''
        class: {}
        dim_data = {}, dim_hash = {}, N_winner = {}
        sparsity_W = {}
        '''.format(self.__class__.__name__, *self.W_data2hash.shape, self.N_winner, np.mean(self.W_data2hash))
        return msg

    def hashing(self, data):
        prehash = np.matmul(np.atleast_2d(data), self.W_data2hash)
        idx_winner = np.argsort(prehash, axis=1)[:, -self.N_winner:]
        hash = np.zeros_like(prehash)
        for h, i in zip(hash, idx_winner): h[i] = 1
        return hash


class bloom_filter:
    def __init__(self, rule, dim_hash, dim_val):
        self.rule = rule
        if rule == 'classic':
            self.W_hash2val = np.zeros((dim_hash, dim_val))
            self.learning = self._learning_classic
        else:
            self.W_hash2val = np.ones((dim_hash, dim_val))
            if rule == 'decay':
                self.learning = self._learning_decay
            else:
                raise ValueError("Unknown learning rule!")

    def __str__(self):
        msg = '''
        class: {}
        rule: {}
        dim_hash = {}, dim_val = {}
        '''.format(self.__class__.__name__, self.rule, *self.W_hash2val.shape)
        return msg

    def evaluating(self, hash, v_max):
        vals = np.divide(np.matmul(np.atleast_2d(hash), self.W_hash2val).T, v_max).T
        return vals

    def _learning_classic(self, hash, rate):
        self.W_hash2val += np.multiply(np.atleast_2d(hash).T, rate)

    def _learning_decay(self, hash, rate):
        self.W_hash2val *= 1 - np.multiply(np.atleast_2d(hash).T, rate)



if __name__ == '__main__':
    point = np.array([[1, 0], [-2, 0], [2, 3]])
    lsh = random_projection_hash(2, 5, 3)
    print(lsh)
    print(lsh.hashing(point))

    mb = k_winners_take_all_hash(3, 10, 0.5, 3)
    print(mb)
    hash_mb = mb.hashing(point.T)

    print(hash_mb)

    bf = bloom_filter('decay', 10, 1)
    print(bf)
    bf.learning(np.random.rand(10), 0.4)
    print(bf.evaluating(hash_mb, 3))