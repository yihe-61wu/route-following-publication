import numpy as np
import cv2

from igibson_api import BaseRobot, quat2axisangle
from robot_sensory import *

from utils import pairwise_distances


def angular2linear(va, vl2_max, inertia_constant=0.3):
    vl2 = vl2_max - inertia_constant * (va ** 2)
    vl = np.sqrt(vl2) if vl2 > 0 else 0
    return vl

def find_angle(a, b, c):
    return np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

def find_side(A, a, b):
    ratio = np.sin(A) / a
    B = np.arcsin(b * ratio)
    # generally 2 solutions, but always one solution here as a > b
    C = np.pi - A - B
    c = np.sin(C) / ratio
    return c


class PurePursuitSteering:
    def __init__(self,
                 robot:BaseRobot,
                 v_linear,
                 lookaround,
                 rot_coeff=5,
                 noise=0):
        self.wheel_axle_length = robot.wheel_axle_length
        self.vl2max = v_linear ** 2
        self.la = lookaround
        self.rot_coeff = rot_coeff
        self.noise = noise
        self.wpts = np.empty((0, 2))
        self.unit_orn = np.empty((0, 2))
        self.last_pursuit = None

    def record_waypoint(self, pos):
        self.wpts = np.concatenate((self.wpts, pos), axis=0)

    def localise_self(self, pos):
        return pos

    def pose2control(self, pos, orn):
        # OracleBasedOdometrySensor.sense_orientation() has been changed
        # output off by pi/2
        # this function needs investigation
        dists = pairwise_distances(pos, self.wpts).flatten()
        idx_inradius = np.argwhere(dists <= self.la).flatten()
        if idx_inradius.size > 0:
            idx_inpursuit = idx_inradius[-1]
            idx_outpursuit = idx_inpursuit + 1
            if idx_outpursuit == self.wpts.shape[0]:
                xy_pursuit = self.wpts[-1]
            else:
                xy_A = self.wpts[idx_inpursuit]     # A: furtherst inradius waypoint
                xy_B = self.wpts[idx_outpursuit]    # B: the next waypoint of A
                v_AB = xy_B - xy_A
                d_a = dists[idx_outpursuit]           # distance between BC
                d_b = dists[idx_inpursuit]            # distance between AC
                d_c = np.linalg.norm(v_AB)            # distance between AB
                ang_A = find_angle(d_a, d_b, d_c)
                d_t = find_side(ang_A, self.la, d_b)
                xy_pursuit = xy_A + v_AB / d_c * d_t
                # the point is the intersection of lookaround circle and line segment AB
        else:
            xy_pursuit = self.wpts[0] if self.last_pursuit is None else self.last_pursuit
        diff_pos = (xy_pursuit - pos).flatten()
        angle2pursuit = (np.arctan2(*np.flip(diff_pos)) - orn) % (2 * np.pi) - np.pi
        va = np.arctan(np.sin(angle2pursuit) * 2 * self.wheel_axle_length / self.la) * self.rot_coeff
        vl = angular2linear(va, self.vl2max)
        return vl, va


class LateralNoveltySteering:
    def __init__(self, v_linear_max, rot_coeff=10, vl_ctrl='kinetic'):
        self.vl_ctrl = vl_ctrl
        self.vl = v_linear_max
        self._vl2max = self.vl ** 2
        self.c_rot = rot_coeff

    def novelty2control(self, nov_l, nov_r):
        va = (nov_l - nov_r) * self.c_rot
        # va = (nov_l / nov_r - 1) * self.c_rot
        if self.vl_ctrl == 'kinetic':
            vl = angular2linear(va, self._vl2max)
        elif self.vl_ctrl == 'differential':
            vl = (1 - nov_l - nov_r) * self.vl
        return va, vl


class LateralFamiliaritySteering:
    def __init__(self, v_linear, rot_coeff=10):
        self.vl = v_linear
        self._vl2max = self.vl ** 2
        self.c_rot = rot_coeff

    def familiarity2control(self, fam_l, fam_r):
        fam_sum = fam_r + fam_l
        fam_dif = fam_r - fam_l
        if fam_sum == 0:
            va = 1 if fam_dif >= 0 else -1
        else:
            va = self.c_rot * fam_dif / fam_sum
        vl = angular2linear(va, self._vl2max)
        return va, vl


class Lateral3viewFamiliaritySteering(LateralFamiliaritySteering):
    def familiarity2control(self, fam_l, fam_m, fam_r):
        fam_sum = fam_r + fam_m + fam_l
        fam_dif = fam_r - fam_l
        fam_min = np.min([fam_l, fam_m, fam_r])
        fam_max = np.max([fam_l, fam_m, fam_r])
        if fam_m == fam_max:
            vl = self.vl
            va = fam_dif / (fam_m - fam_min)
        else:
            vl = 0
            va = np.sign(fam_dif)
        return va, vl


class LateralFamiliaritySteeringNotNormalised:
    def __init__(self, v_linear, rot_coeff=1, noise_lv=0):
        self.vl = v_linear
        self._vl2max = self.vl ** 2
        self.c_rot = rot_coeff
        self.noise = noise_lv

    def familiarity2control(self, fam_l, fam_r):
        fam_sum = fam_r + fam_l
        fam_dif = fam_r - fam_l
        if fam_sum == 0:
            va = 1 if fam_dif >= 0 else -1
        else:
            va = self.c_rot * fam_dif
        vl = angular2linear(va, self._vl2max)
        return va, vl


class KlinokinesisSteering:
    def __init__(self, v_linear, rot_coeff=0.3, switch_step=5):
        # switch_step cannot be too small
        self.vl = v_linear
        self._vl2max = self.vl ** 2
        self.c_rot = rot_coeff
        self.fam_max = 0
        self.switch_step = switch_step
        self.clock = 0

    def familiarity2control(self, fam):
        fam_norm = fam / self.fam_max * 2 - 1
        # after learning of a single view any input yields a similarity value near half of the similarity of the learned view
        nov_norm = np.maximum(1 - fam_norm, 0)
        pm = int(self.clock < self.switch_step) * 2 - 1
        va = self.c_rot * nov_norm * pm
        vl = angular2linear(va, self._vl2max)
        self.clock = (self.clock + 1) % (self.switch_step * 2)
        return va, vl

