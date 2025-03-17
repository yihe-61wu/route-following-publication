import numpy as np
import scipy.spatial as sp_spatial

# to be deprecated
def pairwise_distances(Xs, Ys):
    return sp_spatial.distance.cdist(Xs, Ys).T


# to be deprecated
def points2curve(points, curve):
    dists = pairwise_distances(points, curve)
    p2c = np.min(dists, axis=0)
    p2c_complex = [point2segments(p, curve) for p in points]
    diff = np.abs(p2c_complex - p2c)
    # print(np.mean(diff), np.max(diff))
    return p2c

# to be deprecated
def curve2curve(curve_X, curve_Y, metric):
    dists = points2curve(curve_X, curve_Y)
    if metric == 'last':
        dist = dists[-1]
    elif metric == 'max':
        dist = np.max(dists)
    elif metric == 'avg':
        dist = np.mean(dists)
    elif metric == 'sym-avg':
        dist = (curve2curve(curve_X, curve_Y, 'avg') + curve2curve(curve_Y, curve_X, 'avg')) / 2
    return dist


def point2segments(point, segment_nodes):
    point = np.atleast_2d(point)
    nodes = np.array(segment_nodes)
    point2nodes = sp_spatial.distance.cdist(point, nodes).flatten()
    n_node = np.shape(nodes)[0]
    # find the endpoint indices of the segment closest to the point
    idx_mindist = np.argmin(point2nodes)
    if idx_mindist == 0:
        idx_neighbour = 1
    elif idx_mindist == n_node - 1:
        idx_neighbour = n_node - 2
    else:
        idx_prev, idx_post = idx_mindist - 1, idx_mindist + 1
        idx_neighbour = np.where(point2nodes[idx_prev] <= point2nodes[idx_post], idx_prev, idx_post)
    seg_start, seg_end = nodes[idx_mindist], nodes[idx_neighbour]
    # vector dot product for determining if closest point on segment
    vec1 = point - seg_start
    vec2 = seg_end - seg_start
    dotprod = np.dot(vec1, vec2)
    if dotprod <= 0:
        p2s = point2nodes[idx_mindist]
    else:
        projection = seg_start + vec2 * dotprod / np.linalg.norm(vec2 ** 2)
        p2s = np.linalg.norm(projection - point)
    return p2s


def segment2path(segment_points, path_nodes):
    points2path = [point2segments(p, path_nodes) for p in segment_points]
    return np.mean(points2path)

def segpath2path(segment_points, path_nodes):
    segment_length = np.linalg.norm(segment_points[1:] - segment_points[:-1], axis=1)
    segment_length = np.insert(segment_length, [0, segment_length.size], 0)
    segpath_totallen = np.sum(segment_length)
    weighted_dist = 0
    points2path = []
    for pidx, p in enumerate(segment_points):
        p2path_dist = point2segments(p, path_nodes)
        points2path.append(p2path_dist)
        p_weight = segment_length[pidx] + segment_length[pidx + 1]
        weighted_dist += p2path_dist * p_weight
    if segpath_totallen > 0:
        dissim = weighted_dist / segpath_totallen / 2
    else:
        dissim = p2path_dist
    return dissim, np.array(points2path)


def radian_absolute_difference(rad1, rad2):
    # this is the absolute angular difference
    diff = (rad1 - rad2) % (np.pi * 2)
    return np.minimum(diff, np.pi * 2 - diff)


def radian_difference_a2b(radian_a, radian_b):
    rad_diff = (radian_b - radian_a) % (np.pi * 2)
    if rad_diff > np.pi: rad_diff -= np.pi * 2
    return rad_diff


def add_allo_ego(x, y, rad, dist):
    new_x = x + np.cos(rad) * dist
    new_y = y + np.sin(rad) * dist
    return new_x, new_y


class queue:
    def __init__(self, queue_length, init_val=None):
        self.length = queue_length
        self.init_val = init_val
        self.reset()

    def __getitem__(self, index):
        return self.memory[index]

    def reset(self):
        self.memory = []
        if self.init_val is not None:
            self.update(self.init_val)

    def update(self, new_item):
        for _ in range(self.length + 1 - len(self.memory)):
            self.memory.append(new_item)
        self.memory.pop(0)
        return self.memory



def spl(path, goal, shortest_dist, goal_tol=0.2):
    success = np.linalg.norm(path[-1] - goal) <= goal_tol
    if success:
        path_len = np.sum(np.linalg.norm(path[:-1] - path[1:], axis=1))
        spl = shortest_dist / np.maximum(shortest_dist, path_len)
    else:
        spl = 0
    return spl