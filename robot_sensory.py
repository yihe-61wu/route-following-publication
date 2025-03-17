import numpy as np
import cv2
from igibson_api import RobotSensorBase
from image_preprocessing import apply_lateral_inhibition
from neuro_utils import random_projection_hash


import matplotlib.pyplot as plt


class OracleBasedOdometrySensor(RobotSensorBase):
    def __init__(self,
                 simulator,
                 robot,
                 update_timestep=1,
                 noise=0.0):
        super().__init__(simulator, robot, noise)
        self.update_timestep = np.maximum(update_timestep, self.oracle.simulator.render_timestep)

    def sense_position(self):
        pos = self.oracle.get_position()
        sigma = self.oracle.get_speed_linear() * self.noise * self.update_timestep
        if sigma > 0: pos = np.random.normal(pos, sigma)
        return pos

    def sense_orientation(self):
        orn = self.oracle.get_orientation()
        sigma = np.abs(self.oracle.get_speed_angular()) * self.noise * self.update_timestep
        if sigma > 0: orn = np.random.vonmises(orn, np.sqrt(1 / sigma))
        return orn % (2 * np.pi)

    def sense_speed_linear(self):
        vl = self.oracle.get_speed_linear()
        if self.noise > 0: vl *= np.exp(np.random.normal(scale=self.noise))
        return vl

    def sense_speed_angular(self):
        va = self.oracle.get_speed_angular()
        if self.noise > 0: va *= np.exp(np.random.normal(scale=self.noise))
        return va


class ConstrainedVisualSensor(RobotSensorBase):
    def __init__(self,
                 simulator,
                 robot,
                 view_HW=(33, 33),
                 grey=True,
                 lateral_inhibition=None,
                 sigma_DoG=(1, 2),
                 blur_boxW=7,
                 preprocess='id',
                 noise=0,
                 **kwargs):
        self.view_HW = view_HW
        self.view_size = np.multiply(*view_HW)
        self.grey = grey
        self.LI = lateral_inhibition
        self.sigma_DoG = sigma_DoG
        self.blur_boxW = blur_boxW
        self.preprocess = preprocess    # 'id', 'ih', 'h' are expected to whiten data the most
        if 'i' in preprocess:   # lateral inhibition by Difference of Gaussians
            self.sigma12 = kwargs['sigma12']
        if 'h' in preprocess:   # use random projection LSH instead of dimension reduction
            self.lsh = random_projection_hash(np.prod([simulator.image_height, simulator.image_width]), self.view_size)
        super().__init__(simulator, robot, noise / np.sqrt(np.multiply(*view_HW)))

    def old_rgbframe2view(self, rgbframe):
        view = cv2.resize(rgbframe, self.view_HW)
        if self.grey:
            view = cv2.cvtColor(view, cv2.COLOR_RGB2GRAY)
        if self.LI is None:
            if self.blur_boxW > 1:
                view = cv2.blur(view, (self.blur_boxW, self.blur_boxW))
        elif self.LI == 'DoG':
            view_LI = apply_lateral_inhibition(view, *self.sigma_DoG)
        else:
            raise NotImplementedError('Unknow eye model!')
        return view

    def rgbframe2view(self, rgbframe):
        view = cv2.cvtColor(rgbframe, cv2.COLOR_RGB2GRAY)   # hardly processing color channels differently
        for pp in self.preprocess:
            view = self._preprocess_view(view, pp)
        return view

    def _preprocess_view(self, view, transform):
        if transform == 'i':
            img = apply_lateral_inhibition(view, *self.sigma12)
        elif transform == 'd':
            img = cv2.resize(view, self.view_HW)
        elif transform == 'h':
            img = self.lsh.hashing(view.flatten())
        return img

    def get_view(self):
        return self.rgbframe2view(self.oracle.get_frame_rgb())


class LateralisedVisualSensor(ConstrainedVisualSensor):
    def __init__(self,
                 simulator,
                 robot,
                 view_HW=(33, 33),
                 eye_W=22,
                 grey=True,
                 lateral_inhibition=None,
                 sigma_DoG=(1, 2),
                 blur_boxW=7,
                 preprocess='id',
                 noise=0,
                 **kwargs):
        super().__init__(simulator, robot, view_HW, grey, lateral_inhibition, sigma_DoG, blur_boxW, preprocess, noise, **kwargs)
        self.eye_W = eye_W if eye_W < view_HW[1] else view_HW[1]
        self.eye_size = self.view_HW[0] * self.eye_W
        self.eye_lb_max = self.view_HW[1] - self.eye_W
        self.mid_eye_lb = self.eye_lb_max // 2

    def get_1eye_view(self, offset=0):
        if isinstance(offset, int):
            eye_lb = np.clip(self.mid_eye_lb + offset, 0, self.eye_lb_max)
        elif offset == 'l':
            eye_lb = 0
        elif offset == 'r':
            eye_lb = self.eye_lb_max
        view = self.get_view()[:, eye_lb : eye_lb + self.eye_W]
        if self.noise > 0: view = np.multiply(view, np.random.normal(np.ones_like(view), scale=self.noise))
        return view


class ProximityCollisionSensor(RobotSensorBase):
    def __init__(self,
                 simulator,
                 robot,
                 force_threshold=100,
                 depth_threshold=1,
                 depth_vertical_range_percentile=(45, 65),
                 noise=0.0):
        super().__init__(simulator, robot, noise)
        self.thre_force = force_threshold
        self.thre_depth = depth_threshold * robot.scale * 0.559 / 2 * np.sqrt(2)
        # threshold * scale * freight radius * sqrt(2) - due to 90 fov
        # when depth_threshold=1, in tests by examine_obstacle.py on 12/06/2024, only 1 epoch where detected as proximal
        self.frame_hw = simulator.image_height, simulator.image_width
        self.dvr_idx = np.multiply(self.frame_hw, depth_vertical_range_percentile) // 100 + [0, 1]

    def feel_depth(self):
        # old method, treat front depth as a scalar
        depth_frame = self.get_depth_frame()
        depth_horizon = np.nanmean(depth_frame, axis=0)
        depth_horizon[np.isnan(depth_horizon)] = 0
        depth = np.min(depth_horizon)
        # depth = np.min(depth_frame)   # not enough to prevent collision
        if self.noise > 0: depth *= np.random.lognormal(0, self.noise)
        return depth

    def get_depth_frame(self):
        depth_view = self.oracle.get_frame_depth()
        view_b = np.where(depth_view == 0, 1, 0)
        view_g = np.where(depth_view >= self.thre_depth, np.log(depth_view), 0)
        depth_view[depth_view==0] = np.nan     # by default, dist to background image is 0, causing unwanted behaviour
        # depth_view[depth_view>=self.thre_depth] = np.nan     # the antannae-like depth has a very limited range
        depth_frame = depth_view[self.dvr_idx[0] : self.dvr_idx[1]]
        view_r = np.where(depth_view < self.thre_depth, 1 - depth_view, 0)
        img_8u = cv2.convertScaleAbs(np.stack((view_b, view_g, view_r), axis=2), alpha=255.0)
        cv2.imshow('depth', img_8u)
        cv2.imshow('depth cut {} to {}'.format(self.dvr_idx[0], self.dvr_idx[1]), img_8u[self.dvr_idx[0] : self.dvr_idx[1]])
        # plt.plot(np.mean(depth_view, axis=1))
        return depth_frame

    def get_depth_horizon(self):
        depth_frame = self.get_depth_frame()
        depth_horizon = np.nanmin(depth_frame, axis=0)
        return depth_horizon

    def feel_force(self, magnitude_only=True):
        force = self.oracle.get_force_horizon(magnitude_only)
        if self.noise > 0: force *= np.random.lognormal(0, self.noise)
        return force

    def feel_proximity(self):
        proximal = self.feel_depth() <= self.thre_depth
        return proximal

    def feel_collision(self):
        collided = self.feel_force() >= self.thre_force or self.feel_depth() <= 0
        return collided

