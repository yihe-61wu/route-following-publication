import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS, BaseRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.stadium_scene import StadiumScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
from igibson.utils.assets_utils import get_ig_scene_path
from igibson.utils.transform_utils import quat2axisangle, axisangle2quat

import pybullet as p
import numpy as np
import scipy as sp
import cv2
import os
from datetime import datetime
from sys import platform

from utils import pairwise_distances


class SimulationOrganisor:
    def __init__(self, scene_name, robot_name, headless, **kwargs):
        self.scene_name = scene_name
        self.robot_name = robot_name
        self.obstacles = {}
        self.headless = headless
        self.stamp = self._create_stamp()
        self.mindist_start2goal = kwargs['mindist_start2goal'] if 'mindist_start2goal' in kwargs.keys() else 3
        self.pos_start = None
        self.pos_goal = None
        self.optimal_route = None
        self.add_moving_obstacle = False

    def _create_stamp(self):
        now = datetime.now()
        dt_string = now.strftime("_%Y%m%d_%H%M%S_")
        stamp = dt_string + self.scene_name
        return stamp

    def prepare_robot(self):
        self.robot, self.robot_name = self._config_robot(self.robot_name)
        return self.robot

    def _config_robot(self, robot_name):
        robot_config_file = robot_name + '.yaml'
        config = parse_config(os.path.join(igibson.root_path, 'yihe', 'yaml_robots', robot_config_file))
        robot_config = config["robot"]
        robot_name = robot_config.pop("name")
        robot = REGISTERED_ROBOTS[robot_name](**robot_config)
        if robot_name == 'Freight':
            self.robot_radius = 0.559 * robot.scale / 2
        else:
            self.robot_radius = robot.scale
        return robot, robot_name

    def prepare_dynamical_obstacles(self, robot_list, init_loc):
        self.add_moving_obstacle = True
        for robot_name, robot_pos in zip(robot_list, init_loc):
            obstacle, robot_name = self._config_robot(robot_name)
            obstacle_name = 'dyob_' + robot_name
            self.obstacles[obstacle_name] = obstacle, robot_pos
        return self.obstacles

    def prepare_scene(self,
                      load_object_categories=None,
                      texture_randomization=False,
                      trav_map_resolution=0.01):
        if self.scene_name == 'random':
            dir_scene = os.path.join(igibson.ig_dataset_path, 'scenes')
            ls_scene = os.listdir(dir_scene)
            two_scene_name = np.random.choice(ls_scene, 2, replace=False)
            self.scene_name = two_scene_name[1] if two_scene_name[0] == 'background' else two_scene_name[0]
            # rewritng self.scene_name is not ideal: the specified scene_name and the actual scene_name don't have to
            # be identical; this also causes troubles in fix_route, and random route generation
        if self.scene_name == 'empty':
            self.scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
        elif self.scene_name == 'stadium':
            self.scene = StadiumScene()
        else:
            trav_map_type = 'with_obj' if load_object_categories is None else 'with_no_obj'
            trav_map_erosion = np.ceil(2 * self.robot.wheel_axle_length / trav_map_resolution).astype(int)
            # hotfix: coefficient 2 indicates the footprint is larger than wheel axle length
            self.scene = InteractiveIndoorScene(self.scene_name,
                                                trav_map_type=trav_map_type,
                                                trav_map_resolution=trav_map_resolution,
                                                trav_map_erosion=trav_map_erosion,
                                                load_object_categories=load_object_categories,
                                                should_open_all_doors=True,
                                                texture_randomization=texture_randomization)
            #, object_randomization=True)
            self.scene_name = self.scene.scene_id
        return self.scene

    def prepare_simulator(self):
        if self.headless:
            s_mode = 'headless'
        else:
            s_mode = 'gui_interactive'
        if self.scene_name in ('empty', 'stadium'):
            settings = MeshRendererSettings(enable_shadow=True, msaa=True, texture_scale=0.5)
        else:
            hdr_texture1 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
            hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
            light_map = os.path.join(get_ig_scene_path(self.scene_name), "layout", "floor_lighttype_0.png")
            background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")
            settings = MeshRendererSettings(
                env_texture_filename=hdr_texture1,
                env_texture_filename2=hdr_texture2,
                env_texture_filename3=background_texture,
                light_modulation_map_filename=light_map,
                enable_shadow=True,
                msaa=True,  # reduce edges
                enable_pbr=True,  # enable realistic shadow
                skybox_size=36.0,
                light_dimming_factor=1.0,  # default is 1
                texture_scale=0.5 if platform == "darwin" else 1.0,  # Reduce scale if in Mac
            )
        self.simulator = Simulator(mode=s_mode, image_width=256, image_height=256, rendering_settings=settings)
        return self.simulator

    def prepare_birdviewer(self, direction):
        if not self.headless:
            self.birdviewer = self.simulator.viewer
            self.birdviewer.initial_view_direction = np.array(direction)
            self.birdviewer.initial_pos = self.init_robot_pos_orn[0] - self.birdviewer.initial_view_direction
            self.birdviewer.reset_viewer()

    def _init_robot_pos(self, check_step):
        if self.scene_name in ('empty', 'stadium'):
            position = [0, 0, 0]
            room = None
            self.init_robot_pos_orn = position, [0, 0, 0, 1]
        else:
            if self.fix_route:
                # assert self.scene_name == 'Rs_int'
                if self.scene_name == 'Rs_int':
                    position = [-1.0, -0.2, 0.0]
                    room = "living room"
                    self.manual_va = np.concatenate((np.full(15, -0.05), np.full(60, 0.1), np.full(25, -0.1), np.full(25, 0.05)))
                    self.init_robot_pos_orn = position, [0, 0, 0, 1]
                    self.train_time = 125
                elif self.scene_name == 'Pomaria_1_int':
                    position = [-6, 2, 0] #self.scene.get_random_point()[1] #[-12, 2.2, 0]
                    room = "living room"
                    self.manual_va = np.full(300, 0)
                    self.init_robot_pos_orn = position, axisangle2quat([0, 0, np.random.rand() * np.pi * 2])
                elif self.scene_name == 'Ihlen_0_int':
                    position = [2.5, 9, 0]
                    # position = [-4.5, 9, 0]
                    room = "living room"
                    self.manual_va = np.full(250, 0)
                    self.init_robot_pos_orn = position, axisangle2quat([0, 0, np.pi])
                    # self.init_robot_pos_orn = position, axisangle2quat([0, 0, 0])
                    self.train_time = 125
            else:
                path_node = 0
                while path_node < 4:    # too short path doesn't permit interpolate; longer path better for experiment
                    path = self._rand_train_route(check_step)
                    path_node = path.shape[0]
                spl = sp.interpolate.make_interp_spline(np.arange(path.shape[0]), path)
                self.train_route = spl(np.linspace(0, path.shape[0], self.train_time))
                self.manual_po = self._compute_manual_pos_orn()
                position = np.concatenate((self.train_route[0], [0]))
                room = self.scene.get_room_instance_by_point(position[:2])
                self.init_robot_pos_orn = position, self.manual_po[1][0]
        return position, room

    def _compute_manual_pos_orn(self):
        train_pos = np.zeros((self.train_time, 3))
        train_orn = np.zeros((self.train_time, 4))
        for t in range(self.train_time):
            train_pos[t, :2] = self.train_route[t]
            if t < self.train_time - 1:
                orn_vec = self.train_route[t + 1] - self.train_route[t]
                orn_rad = np.arctan2(orn_vec[1], orn_vec[0])
                train_orn[t] = axisangle2quat([0, 0, orn_rad])
            else:
                train_orn[t] = train_orn[t - 1]
        return train_pos, train_orn


    def _rand_train_route(self, check_step):
        dist_max = 0
        n_while = 0
        while dist_max < (self.mindist_start2goal * 0.5 ** (n_while // check_step)):
            n_while += 1
            # generate 3 random points
            points, rooms = [], []
            for _ in range(3):
                pos, room = self._rand_valid_robot_pos(check_step)
                points.append(pos[:2])
                rooms.append(room)
            pts_abc = np.reshape(points, (3, 2))
            # pick the 2 shortest sides as the stat and end
            dist_abc = pairwise_distances(pts_abc, pts_abc)
            dist_ab_bc_ca = dist_abc[1, 0], dist_abc[2, 1], dist_abc[0, 2]
            idx_0, idx_1 = np.argsort(dist_ab_bc_ca)[:2]
            idx_max = np.argmax(dist_ab_bc_ca)
            dist_max = dist_ab_bc_ca[idx_max]
        path = np.empty((0, 2))
        for i in (0, 1):
            start = pts_abc[idx_max - i]
            end = pts_abc[idx_max - i - 1]
            shortest_path = self.scene.get_shortest_path(0, start, end, True)
            path = np.concatenate((path, shortest_path[0]), axis=0)
        return path

    def _rand_valid_robot_pos(self, collision_check_step):
        collided = True
        while collided:
            pos = self.scene.get_random_point()[1]
            self.robot.set_position(pos)
            self.robot.reset()
            self.robot.keep_still()
            for t in range(collision_check_step):
                self.simulator.step()
                force = self.oracle.get_force_horizon()
                collided = force >= 10
                if collided: break
        room = self.scene.get_room_instance_by_point(pos[:2])
        return pos, room

    def initialise_simulation(self,
                              fix_route=False,
                              viewer_direction=[0, 0, -1],
                              check_step=10,
                              train_time=120):
        self.simulator.import_scene(self.scene)
        self.simulator.import_object(self.robot)
        self.oracle = IgibsonOracle(self.simulator, self.robot)
        self.fix_route = fix_route
        self.train_time = train_time    # the next line may rewrite self.train_time if fix_route true
        init_robot_pos, init_room = self._init_robot_pos(check_step)
        self.robot.set_position_orientation(*self.init_robot_pos_orn)
        self.prepare_birdviewer(viewer_direction)
        if not self.headless: print("robot standby in {} of {}".format(init_room, self.scene_name))

    def initialise_pointgoal_simulation(self,
                                        max_trial_time,
                                        fix_pos_goal,
                                        add_moving_obstacle=False,
                                        check_step=10,
                                        viewer_direction=[0.15, 0, -1]):
        self.simulator.import_scene(self.scene)
        self.simulator.import_object(self.robot)
        self.oracle = IgibsonOracle(self.simulator, self.robot)
        self.max_trial_time = max_trial_time
        _, init_room = self._init_start_goal(check_step, fix_pos_goal)
        self.init_robot_pos_orn = self.pos_start, axisangle2quat([0, 0, np.random.rand() * np.pi * 2])
        self.robot.set_position_orientation(*self.init_robot_pos_orn)
        self.prepare_birdviewer(viewer_direction)
        if not self.headless: print("robot standby in {} of {}".format(init_room, self.scene_name))
        self.optimal_route = self.scene.get_shortest_path(0, self.pos_start[:2], self.pos_goal[:2], True)
        ###### moving obstacles
        if self.add_moving_obstacle:
            for obstacle, init_pos in self.obstacles.values():
                self.simulator.import_object(obstacle)
                obstacle.set_position(init_pos)
                obstacle.apply_action([-0.2, -0.1])

    def initialise_pointgoal_simulation2(self,
                                        max_trial_time,
                                        fix_pos_goal,
                                        check_step=10,
                                        viewer_direction=[0.15, 0, -1]):
        self.simulator.import_object(self.robot)
        self.oracle = IgibsonOracle(self.simulator, self.robot)
        self.max_trial_time = max_trial_time
        _, init_room = self._init_start_goal(check_step, fix_pos_goal)
        self.init_robot_pos_orn = self.pos_start, axisangle2quat([0, 0, np.random.rand() * np.pi * 2])
        self.robot.set_position_orientation(*self.init_robot_pos_orn)
        self.prepare_birdviewer(viewer_direction)
        if not self.headless: print("robot standby in {} of {}".format(init_room, self.scene_name))

    def _init_start_goal(self, check_step, fix_pos_goal=None):
        if fix_pos_goal is not None:
            self.pos_start, self.pos_goal = fix_pos_goal
            room_start = 'room'
        else:
            dist_start2goal = 0
            n_while = 0
            while dist_start2goal < (self.mindist_start2goal * 0.5 ** (n_while // check_step)):
                n_while += 1
                self.pos_start, room_start = self._rand_valid_robot_pos(check_step)
                self.pos_goal, _ = self._rand_valid_robot_pos(check_step)
                dist_start2goal = np.linalg.norm(self.pos_goal - self.pos_start)
        return self.pos_start, room_start

    def reinit_robot_pose(self, rand_orn=False, cooling_time=1):
        init_pos, init_orn = self.init_robot_pos_orn
        if rand_orn:
            init_angle = quat2axisangle(init_orn)
            rand_az = np.random.rand() * 2 * np.pi - np.pi
            rand_angle = np.append(init_angle[:2], rand_az)
            rand_quat = axisangle2quat(rand_angle)
            pos, orn = init_pos, rand_quat
        else:
            pos, orn = init_pos, init_orn
        self.robot.reset()
        self.robot.set_position_orientation(pos, orn)
        self.robot.keep_still()
        [self.simulator.step() for _ in range(int(cooling_time // self.simulator.render_timestep))]
        if self.scene_name not in ('empty', 'stadium'): self.scene.reset_scene_objects()
        ###### moving obstacles
        if self.add_moving_obstacle:
            for obstacle, init_pos in self.obstacles.values():
                obstacle.set_position(init_pos)
                obstacle.apply_action([0.2, -0.1])

    def manual_action(self, step, vl=0.5, va=-0.01):
        if self.fix_route:
            va = self.manual_va[step]
            if self.scene_name in ('Rs_int', 'Ihlen_0_int'):
                vl = 0.5
            elif self.scene_name == 'Pomaria_1_int':
                vl = 1
        return vl, va


class IgibsonOracle:
    def __init__(self,
                 simulator:Simulator,
                 robot:BaseRobot):
        self.cameras = simulator.renderer.render_robot_cameras
        self.robot = robot
        self.simulator = simulator

    def get_frame_rgb(self):
        return self.cameras(modes=("rgb"), cache=False)[0][:, :, :3]

    def get_frame_depth(self):
        return np.linalg.norm(self.cameras(modes=("3d"))[0][:, :, :3], axis=2)

    def get_force_horizon(self, magnitude_only=True):
        sum_horizon_force = 0
        for collision in p.getContactPoints(bodyA=self.robot.get_body_ids()[0], linkIndexA=-1):
            contact_normal = collision[7]
            normal_force = collision[9]
            horizon_force_vec = normal_force * np.array(contact_normal)[:2]
            sum_horizon_force += horizon_force_vec
        if magnitude_only: sum_horizon_force = np.linalg.norm(sum_horizon_force)
        return sum_horizon_force

    def get_position(self, verbose=False):
        pos = self.robot.get_position()
        if not verbose: pos = pos[:2]
        return pos

    def get_orientation(self, verbose=False):
        orn = quat2axisangle(self.robot.get_orientation())
        if not verbose: orn = orn[2]
        return orn

    def get_speed_linear(self):
        return np.linalg.norm(self.robot.get_linear_velocity()[:2])

    def get_speed_angular(self):
        return -self.robot.get_angular_velocity()[2]
        # negative sign needed due to unclear API feature/bug


class RobotSensorBase:
    def __init__(self,
                 simulator:Simulator,
                 robot:BaseRobot,
                 noise=0.0):
        self.oracle = IgibsonOracle(simulator, robot)
        self.noise_sensor = np.maximum(0.0, noise)
        self.set_noise(True)

    def set_noise(self, switch_on=True):
        if switch_on:
            self.noise = self.noise_sensor
        else:
            self.noise = 0


class NoisyDDController:
    def __init__(self,
                 robot:BaseRobot=None,
                 noise=0.0,
                 linear_noise=None,
                 angular_noise=None):
        self.noise = np.maximum(0.0, noise)
        self.vl_noise, self.va_noise = [self._init_noise_level(vn)
                                        for vn in (linear_noise, angular_noise)]
        self.robot_loaded = robot is not None
        if self.robot_loaded:
            self.robot = robot
            self.max_vlva = robot._controllers['base'].command_output_limits[1]
        else:
            self.max_vlva = 1, 1
            msg = "linear noise: {}\nangular noise: {}".format(self.vl_noise, self.va_noise)
            print(msg)

    def _init_noise_level(self, noise):
        return self.noise if noise is None else np.maximum(0.0, noise)

    def apply_action(self, vl, va, noisefree):
        action = []
        for v, noise in zip((vl, va), (self.vl_noise, self.va_noise)):
            if noise > 0 and not noisefree: v *= np.random.lognormal(0, noise)
            action.append(np.clip(v, -1, 1))
        if self.robot_loaded: self.robot.apply_action(action)
        return np.multiply(action, self.max_vlva)


if __name__ == '__main__':
    scene_name = 'random'
    robot_name = 'scaled_freight'
    headless = False
    for _ in range(1):
        simulation = SimulationOrganisor(scene_name, robot_name, headless)
        robot = simulation.prepare_robot()
        scene = simulation.prepare_scene()
        s = simulation.prepare_simulator()
        simulation.initialise_simulation()
        s.disconnect()
