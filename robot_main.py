import numpy as np

from mushroom_body import SingleOutputMB, MultiOutputMB, MOMBnovelty
from robot_steering import *

from robot_sensory import ProximityCollisionSensor
from igibson_api import NoisyDDController
from utils import add_allo_ego


class SensoriMotorModel:
    def __init__(self, sensors):
        self.sensors = tuple(sensors)

    def perceive(self):
        raise NotImplementedError()

    def autoride(self):
        raise NotImplementedError()

    def noisefree(self, noisefree=False):
        [sensor.set_noise(not noisefree) for sensor in self.sensors]

    def get_noise(self):
        return [sensor.noise for sensor in self.sensors]


class EmbodiedSensoryMotorModel:
    def __init__(self,
                 sensory_motor_model:SensoriMotorModel,
                 controller:NoisyDDController,
                 obstacle_sensor:ProximityCollisionSensor,
                 obstacle_avoidance=True):
        self.model = sensory_motor_model
        self.sensor_obstacle = obstacle_sensor
        self.avoid_obstacle = obstacle_avoidance
        self.controller = controller
        self.last_route_pos = None

    def _override(self, vl, va):
        normal = False
        collided = self.sensor_obstacle.feel_collision()
        proximal = self.sensor_obstacle.feel_proximity()
        if self.avoid_obstacle:
        # maybe better to use directly force and depth
            if collided:
                # vl = 0 # roughly works
                # a more insect-like agent would move backwards if collided
                # the info should be noisy
                vl = -self.sensor_obstacle.oracle.get_speed_linear()
                va = np.sign(-self.sensor_obstacle.oracle.get_speed_angular())
            elif proximal:
                # vl = 0
                # va = np.sign(va)
                depth_horizon = self.sensor_obstacle.get_depth_horizon()
                d_r, d_l = np.sum(depth_horizon[depth_horizon.size//2:]), np.sum(depth_horizon[:depth_horizon.size//2])
                if d_r + d_l <= 0:
                    va = int(d_r >= d_l)
                else:
                    va = (d_r - d_l) / (d_r + d_l) * 10
                vl = 1
                # print('special va', va)
                # to be modularised in an obstacle avoidance class
            else:
                normal = True
        action = vl, va
        return action, collided, proximal, normal

    def control(self, learning, manual_vlva=None, noisefree=False, spinonly=False, va_bias=0):
        self.model.noisefree(noisefree)
        perception = self.model.perceive(learning)
        if manual_vlva is None:
            vl, va = self.model.autoride(*perception)
        else:
            vl, va = manual_vlva
        if spinonly: vl = 0
        (vl_idea, va_idea), collided, proximal, normal = self._override(vl, va)    # normalised between -1 and 1
        if va_bias != 0: va_idea += va_bias
        action_idea = vl_idea, va_idea
        action_real = self.controller.apply_action(*action_idea, noisefree)    # real velocities without normalisation
        # this real velocities are not up to the scale of the robot, and is almost useless compared to oracle
        return action_real, action_idea, perception, collided, proximal, normal

    def learn_pos_orn(self, route_pos, route_orn, noisefree=False, learn=True):
        self.controller.robot.set_position_orientation(route_pos, route_orn)
        if self.last_route_pos is not None:
            vl = np.linalg.norm(route_pos - self.last_route_pos) / self.sensor_obstacle.oracle.simulator.render_timestep
            self.controller.robot.set_velocities([(np.array([vl, 0, 0]), np.array([0, 0, 0]))]) ## wrong but maybe work
        self.model.noisefree(True)
        perception = self.model.perceive(learn)
        self.controller.robot.keep_still()
        self.last_route_pos = route_pos
        return perception


### all the models below should be a child class to a parent class
class PurePursuit(SensoriMotorModel):
    def __init__(self,
                 sensor: OracleBasedOdometrySensor,
                 controller: PurePursuitSteering):
        self.sensor = sensor
        super().__init__([self.sensor])
        self.controller = controller

    def perceive(self, learning):
        pos = [self.sensor.sense_position()]
        if learning:
            self.controller.record_waypoint(pos)
            orn = self.sensor.sense_orientation()
        else:
            pos = self.controller.localise_self(pos)
            orn = self.sensor.sense_orientation()
        return pos, orn

    def autoride(self, pos, orn):
        return self.controller.pose2control(pos, orn)


class NaiveVisualLocalisor(SensoriMotorModel):
    def __init__(self,
                 visual_sensor: ConstrainedVisualSensor,
                 odometry_sensor: OracleBasedOdometrySensor,
                 controller: PurePursuitSteering):
        self.visual_sensor = visual_sensor
        self.odometry_sensor = odometry_sensor
        super().__init__([self.visual_sensor, self.odometry_sensor])
        self.controller = controller
        self.view_memory = []

    def record_view(self, view, pos):
        self.view_memory.append(view)
        self.controller.record_waypoint(pos)

    def localise_self(self, view, pos):
        dissim = np.linalg.norm(np.reshape(self.view_memory - view, (-1, view.size)), axis=1)
        loc_idx = np.argmin(dissim)
        dists = pairwise_distances(pos, self.controller.wpts).flatten()
        true_idx = np.argmin(dists)
        loc_pos = pos + self.controller.wpts[loc_idx] - self.controller.wpts[true_idx]
        return loc_pos

    def perceive(self, learning):
        view = self.visual_sensor.get_view()
        pos = [self.odometry_sensor.sense_position()]
        orn = self.odometry_sensor.sense_orientation()
        if learning:
            self.record_view(view, pos)
        else:
            pos = self.localise_self(view, pos)
        return pos, orn

    def autoride(self, pos, orn):
        return self.controller.pose2control(pos, orn)


class UMBKli(SensoriMotorModel):
    def __init__(self,
                 MB: SingleOutputMB,
                 OL: ConstrainedVisualSensor,
                 CX: KlinokinesisSteering):
        self.OL = OL
        super().__init__([self.OL])
        self.MB = MB
        self.CX = CX

    def view2familiarity(self, learning):
        pn = self.OL.get_view().flatten()
        kc = self.MB.hashing(pn)
        if learning: self.MB.learning(kc)
        fam = self.MB.evaluating(kc)
        self.CX.fam_max = np.maximum(fam, self.CX.fam_max)
        return fam

    def familiarity2control(self, fam):
        va, vl = self.CX.familiarity2control(fam)
        return vl, va

    def perceive(self, learning):
        fam = self.view2familiarity(learning)
        return fam,

    def autoride(self, fam):
        return self.familiarity2control(fam)


class LaMB(SensoriMotorModel):
    def __init__(self,
                 MB_l: SingleOutputMB,
                 MB_r: SingleOutputMB,
                 OL: LateralisedVisualSensor,
                 CX: LateralFamiliaritySteering):
        self.OL = OL
        super().__init__([self.OL])
        self.MB = {'l': MB_l, 'r': MB_r}
        self.CX = CX

    def _view2familiarity_1MB(self, MB_parity, view_offset, learning):
        MB = self.MB[MB_parity]
        pn = self.OL.get_1eye_view(view_offset).flatten()
        kc = MB.hashing(pn)
        fam = np.atleast_1d(MB.evaluating(kc))[0]
        if learning: MB.learning(kc)
        # change order of evaluating and learning should not affect performance a lot
        return fam

    def view2familiarity(self, view_l_offset, view_r_offset, learning):
        fam_l, fam_r = [self._view2familiarity_1MB(p, o, learning) for p, o in zip('lr', (view_l_offset, view_r_offset))]
        return fam_l, fam_r

    def familiarity2control(self, fam_l, fam_r):
        va, vl = self.CX.familiarity2control(fam_l, fam_r)
        return vl, va

    def perceive(self, learning):
        if learning:
            fam_l, fam_r = self.view2familiarity('r', 'l', learning)
        else:
            fam_l, fam_r = self.view2familiarity('l', 'r', learning)
        return fam_l, fam_r

    def autoride(self, fam_l, fam_r):
        return self.familiarity2control(fam_l, fam_r)


class LaMBON(SensoriMotorModel):
    def __init__(self,
                 MB: MultiOutputMB,
                 OL: LateralisedVisualSensor,
                 CX: LateralFamiliaritySteering):
        self.OL = OL
        super().__init__([self.OL])
        self.MB = MB
        self.CX = CX

    def _view2familiarity_1MB(self, view_offset):
        pn = self.OL.get_1eye_view(view_offset).flatten()
        kc = self.MB.hashing(pn)
        fam = np.atleast_1d(self.MB.evaluating(kc))[0]
        return fam, kc

    def view2familiarity(self, learning):
        if learning:
            fam, kc = self._view2familiarity_1MB(0)
            self.MB.learning(kc)
            fam_l, fam_r = fam, fam
        else:
            (fam_l, _), (fam_r, _) = [self._view2familiarity_1MB(lr) for lr in 'lr']
        return fam_l, fam_r

    def familiarity2control(self, fam_l, fam_r):
        va, vl = self.CX.familiarity2control(fam_l, fam_r)
        return vl, va

    def perceive(self, learning):
        fam_l, fam_r = self.view2familiarity(learning)
        return fam_l, fam_r

    def autoride(self, fam_l, fam_r):
        return self.familiarity2control(fam_l, fam_r)


class LaMBON3view(LaMBON):
    def view2familiarity(self, learning):
        if learning:
            fam, kc = self._view2familiarity_1MB(0)
            self.MB.learning(kc)
            fam_l, fam_m, fam_r = fam, fam, fam
        else:
            (fam_l, _), (fam_m, _), (fam_r, _) = [self._view2familiarity_1MB(offset) for offset in ('l', 0, 'r')]
        return fam_l, fam_m, fam_r

    def familiarity2control(self, fam_l, fam_m, fam_r):
        va, vl = self.CX.familiarity2control(fam_l, fam_m, fam_r)
        return vl, va

    def perceive(self, learning):
        fam_l, fam_m, fam_r = self.view2familiarity(learning)
        return fam_l, fam_m, fam_r

    def autoride(self, fam_l, fam_m, fam_r):
        return self.familiarity2control(fam_l, fam_m, fam_r)

class SpatialLaMB(LaMB):
    def __init__(self,
                 MB_l: SingleOutputMB,
                 MB_r: SingleOutputMB,
                 vision: LateralisedVisualSensor,
                 steer: LateralFamiliaritySteering,
                 odometry: OracleBasedOdometrySensor,
                 spatial_oracle=True):
        super().__init__(MB_l, MB_r, vision, steer)
        self.odometry = odometry
        self.spatial_oracle = spatial_oracle
        if spatial_oracle:
            self.pos = []

    def familiarity2control(self, fam_l, fam_r):
        va, vl1 = self.CX.familiarity2control(fam_l, fam_r)
        if self.spatial_oracle:
            dist2route = np.linalg.norm(np.array(self.pos) - self.odometry.sense_position(), axis=1)
            vl2 = 1 - np.min(dist2route)
        if vl2 <= 0.1:
            vl = -0.1
        else:
            vl = vl1 * vl2
            # vl = 1
        print(vl2, vl1, vl)
        return vl, va

    def perceive(self, learning):
        if learning:
            fam_l, fam_r = self.view2familiarity('r', 'l', learning)
            self.pos.append(self.odometry.sense_position())
        else:
            fam_l, fam_r = self.view2familiarity('l', 'r', learning)
        return fam_l, fam_r


class DiMB(SensoriMotorModel):
    def __init__(self,
                 MB_l: MOMBnovelty,
                 MB_r: MOMBnovelty,
                 vision: LateralisedVisualSensor,
                 steer: LateralNoveltySteering,
                 odometry: OracleBasedOdometrySensor=None,
                 init_pose=None):
        self.vision = vision
        self.usecheckpoint = odometry is not None
        if self.usecheckpoint:
            self.odometry = odometry
            self.memo_pos = init_pose[0][:2]
            # self.memo_orn = quat2axisangle(init_pose[1])[2]
            self.memo_target = self.memo_pos
            self.memo_l, self.memo_r = self.memo_pos
        super().__init__([self.vision])
        self.MB = {'l': MB_l, 'r': MB_r}
        self.steer = steer

    def _view2novelty_1MB(self, MB_parity, view_offset, learning):
        MB = self.MB[MB_parity]
        pn = self.vision.get_1eye_view(view_offset).flatten()
        kc = MB.hashing(pn, learning)
        nov = np.atleast_1d(MB.evaluating(kc))[0]
        if learning: MB.learning(kc, nov)
        return nov

    def view2novelty(self, view_l_offset, view_r_offset, learning):
        nov_l, nov_r = [self._view2novelty_1MB(p, o, learning) for p, o in zip('lr', (view_l_offset, view_r_offset))]
        return nov_l, nov_r

    def novelty2control(self, nov_l, nov_r):
        va, vl = self.steer.novelty2control(nov_l, nov_r)
        return vl, va

    def perceive(self, learning):
        if learning:
            nov_l, nov_r = self.view2novelty('r', 'l', learning)
        else:
            nov_l, nov_r = self.view2novelty('l', 'r', learning)
        print('left', nov_l)
        return nov_l, nov_r

    def autoride(self, nov_l, nov_r):
        vl, va = self.novelty2control(nov_l, nov_r)
        if self.usecheckpoint:
            # vl, va = self._2threshold(nov_l, nov_r, vl, va)
            vl, va = self._lrthreshold(nov_l, nov_r, vl, va)
        return vl, va

    def _lrthreshold(self, nov_l, nov_r, vl, va):
        mb_thre = 0.8
        pos, orn = self.odometry.sense_position(), self.odometry.sense_orientation()
        # OracleBasedOdometrySensor.sense_orientation() has been changed
        # output off by pi/2
        # this function needs investigation
        if nov_l <= mb_thre:
            self.memo_l = np.array(add_allo_ego(*pos, orn, (mb_thre - nov_l) * 2))
        if nov_r <= mb_thre:
            self.memo_r = np.array(add_allo_ego(*pos, orn, (mb_thre - nov_r) * 2))
        if nov_l > mb_thre and nov_r > mb_thre:
            target = (self.memo_l + self.memo_r) / 2
            vec_self2target = target - pos
            orn_target = np.arctan2(*np.flip(vec_self2target))
            va = (orn_target - orn) / np.pi
            vl = 0.01
        print('target and vl, va', (self.memo_l + self.memo_r) / 2, vl, va)
        return vl, va


    def _2threshold(self, nov_l, nov_r, vl, va):
        nov_hi, nov_lo = 0.8, 0.8 # need to be optimised
        dist_epoch = vl * self.odometry.update_timestep
        dist_l, dist_r = [dist_epoch * (1 - nov) for nov in (nov_l, nov_r)]
        pos, orn = self.odometry.sense_position(), self.odometry.sense_orientation()
        # OracleBasedOdometrySensor.sense_orientation() has been changed
        # output off by pi/2
        # this function needs investigation
        mat_rot = np.array([[np.cos(orn), -np.sin(orn)], [np.sin(orn), np.cos(orn)]])
        if nov_l < nov_lo and nov_r < nov_lo:
            ego_target = np.array([dist_r - dist_l, dist_r + dist_l]) / np.sqrt(2)
        elif nov_l < nov_lo:
            ego_target = np.array([-dist_l, dist_l]) / np.sqrt(2)
        elif nov_r < nov_lo:
            ego_target = np.array([dist_r, dist_r]) / np.sqrt(2)
        else:
            ego_target = np.zeros(2)
        self.memo_target = pos + np.dot(mat_rot, ego_target)
        if nov_l >= nov_hi or nov_r >= nov_hi:
            vec_self2target = self.memo_target - pos
            orn_target = np.arctan2(*np.flip(vec_self2target))
            va = (orn_target - orn) / np.pi
            vl = 0.01
        print('target and vl, va', self.memo_target, vl, va)
        return vl, va


class SpatialMB(DiMB):
    def __init__(self,
                 MB_l: MOMBnovelty,
                 MB_r: MOMBnovelty,
                 N_spatial_pn,
                 vision: LateralisedVisualSensor,
                 steer: LateralNoveltySteering,
                 odometry: OracleBasedOdometrySensor=None,
                 init_pose=None):
        super().__init__(MB_l, MB_r, vision, steer, odometry, init_pose)
        self.N_spn = N_spatial_pn

    def _view2novelty_1MB(self, MB_parity, view_offset, learning):
        MB = self.MB[MB_parity]
        pn = self.vision.get_1eye_view(view_offset).flatten()
        kc = MB.hashing(pn, learning)
        nov = np.atleast_1d(MB.evaluating(kc))[0]
        if learning: MB.learning(kc, nov)
        return nov


class LaMBmo(LaMB):
    ### LaMB with multi-output MBON ###
    def __init__(self,
                 MB_l: MultiOutputMB,
                 MB_r: MultiOutputMB,
                 vision: LateralisedVisualSensor,
                 motor: LateralFamiliaritySteering,
                 odometry: OracleBasedOdometrySensor=None,
                 MBONnet='random',
                 MBONoutput='customise',
                 **MBONnet_kwargs):
        super().__init__(MB_l, MB_r, vision, motor)
        self.odometry = odometry
        # synonyms (if unnecessary)
        self.vision = self.OL
        # self.motor = self.CX
        ###
        # MBON networks
        # name: (learning, control)
        self._mbonet = MBONnet
        self.MBONnet = {'single': self._mbonet_single,
                        'random': self._mbonet_random,
                        'minfam': self._mbonet_minfam,
                        'periodic': self._mbonet_periodic,
                        'moving range': self._mbonet_moverange,
                        'start range': self._mbonet_startrange,
                        'heading': self._mbonet_heading,
                        'goal angle': self._mbonet_goalangle
                        }[MBONnet]
        # input: familiarity outputs from multiple MBONs
        # output: fam, rates
        #       fam: single familiarity output of this MB
        #       rates: different learning rates of all the MBONs
        self._list_MBoutput = {'max': np.max,
                               'mean': np.mean,
                               'min': np.min,
                               'customise': None
                               }
        self.MBoutput = self._list_MBoutput[MBONoutput]

        self.MBidx = 0
        if self._mbonet == 'periodic':
            self.period = MBONnet_kwargs['period']
            self.clock = 0
        elif self._mbonet in ('moving range', 'start range'):
            self.range = MBONnet_kwargs['range']
            self.origin = self.odometry.sense_position()
        elif self._mbonet in ('heading', 'goal angle'):
            if self._mbonet == 'goal angle':
                self.goal_pos = MBONnet_kwargs['goal']
            elif self._mbonet == 'heading':
                self.primary_heading = np.random.rand() * np.pi * 2
            N_orn = self.MB['l'].N_mbon
            self.mbon_rate = lambda angle: np.cos(angle - np.linspace(0, 2 * np.pi, N_orn, endpoint=False)) + 1 / N_orn

    def reset(self):
        if self._mbonet == 'periodic':
            self.clock = 0
        elif self._mbonet in ('moving range', 'start range'):
            self.origin = self.odometry.sense_position()

    def _view2familiarity_1MB(self, MB_parity, view_offset, learning):
        MB = self.MB[MB_parity]
        pn = self.vision.get_1eye_view(view_offset).flatten()
        kc = MB.hashing(pn)
        fams = np.atleast_1d(MB.evaluating(kc))
        fam, learn_rates = self.MBONnet(fams)
        if learning: MB.learning(kc, learn_rates)
        return np.maximum(fam, 0)

    def _simpler_output(self, fam, multifam):
        if self.MBoutput is not None:
            fam = self.MBoutput(multifam)
        return fam

    def _mbonet_single(self, multifam):
        # this is effectively the single mbon model
        # all MBONs are identical
        rates = np.zeros_like(multifam)
        rates[0] = 1
        fam = multifam[0]
        return fam, rates

    def _mbonet_random(self, multifam):
        # this is probably the base model
        # a random MBON learns
        rates = np.zeros_like(multifam)
        MBidx = np.random.randint(multifam.size)
        rates[MBidx] = 1
        # the same MBON controls
        fam = multifam[MBidx]
        fam = self._simpler_output(fam, multifam)
        return fam, rates

    def _mbonet_minfam(self, multifam):
        # MBON outputing the minimal familarity learns
        rates = np.zeros_like(multifam)
        MBidx = np.argmin(multifam)
        rates[MBidx] = 1
        # a random MBON controls
        fam = multifam[np.random.randint(multifam.size)]
        fam = self._simpler_output(fam, multifam)
        return fam, rates

    def _mbonet_periodic(self, multifam):
        # MBON take turns to learn after a fixed time period
        rates = np.zeros_like(multifam)
        if self.clock >= self.period:
            self.clock -= self.period
            self.MBidx = (self.MBidx + 1) % multifam.size
        rates[self.MBidx] = 1
        self.clock += 0.5    # each lateral MB add 1 epoch
        # a rythmic MBON controls
        fam = multifam[self.MBidx]
        fam = self._simpler_output(fam, multifam)
        return fam, rates

    def _mbonet_moverange(self, multifam):
        # MBONs take turn to learn; an MBON only learns for a fix spatial range in moving
        rates = np.zeros_like(multifam)
        pos_current = self.odometry.sense_position()
        dist_current2origin = np.linalg.norm(pos_current - self.origin)
        if dist_current2origin > self.range:
            self.MBidx = (self.MBidx + 1) % multifam.size
            self.origin = pos_current
        rates[self.MBidx] = 1
        # a MBON after moving a fix distance controls
        fam = multifam[self.MBidx]
        fam = self._simpler_output(fam, multifam)
        return fam, rates

    def _mbonet_startrange(self, multifam):
        # MBONs take turn to learn; an MBON only learns for a fix spatial range wrt start pos
        rates = np.zeros_like(multifam)
        pos_current = self.odometry.sense_position()
        dist_current2origin = np.linalg.norm(pos_current - self.origin)
        MBidx = int(dist_current2origin // self.range % multifam.size)
        rates[MBidx] = 1
        # the MBON of range controls
        fam = multifam[MBidx]
        fam = self._simpler_output(fam, multifam)
        return fam, rates

    def _mbonet_heading(self, multifam):
        # learn & drive according to heading
        orn = self.odometry.sense_orientation() - self.primary_heading
        # OracleBasedOdometrySensor.sense_orientation() has been changed
        # output off by pi/2
        # this function needs investigation
        rates = self.mbon_rate(orn)
        fam = np.dot(multifam, rates)
        fam = self._simpler_output(fam, multifam)
        return fam, rates

    def _mbonet_goalangle(self, multifam):
        # learn & drive according to heading
        vec2goal = self.goal_pos - self.odometry.sense_position()
        goalangle = np.arctan2(vec2goal[1], vec2goal[0])
        rates = self.mbon_rate(goalangle)
        fam = np.dot(multifam, rates)
        fam = self._simpler_output(fam, multifam)
        return fam, rates