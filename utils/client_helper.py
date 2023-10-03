import time
import fsds
import config
import numpy as np
from scipy.spatial.transform import Rotation
from utils.maths import butter_lowpass_filter

class CarState:
    timestamp = None
    car_pos = None
    car_hdg = None
    car_linear_vel = None
    world_linear_vel = None
    car_angular_vel = None
    car_linear_acc = None
    car_angular_acc = None
    car_speed = None
    car_slip = None
    car_steer = None
    car_steer_cmd = None
    car_throttle = None
    car_d_steer = None
    car_rpm = None
    fl_rpm = None
    fr_rpm = None
    rl_rpm = None
    rr_rpm = None


class RefereeState:
    collisions = None
    time_passed_s = None
    lap_number = None
    lap_time = None


class CameraOutput:
    timestamp = None
    car_pos = None
    car_hdg = None
    camera_frame = None



class ClientHelper:
    def __init__(self, logger, controller=False):
        self.logger = logger
        self.client = None
        self.current_map_i = 0
        self.failed_solves = 0
        self.laps_driven = 1
        self.max_speed = None
        self.prev_pos = None
        self.not_moved_steps = 0
        self.start_time = None
        self.delayed_lap_change = False
        self.prev_t = None
        self.dt_history = None
        self.w_history = np.zeros(50)
        self.last_steer = 0
        self.last_throttle = 0
        self.last_d_steer = 0

        self.laps_driven = 1
        self.start_time = None
        self.delayed_lap_change = False

        self.logger.info("Trying to connect client to sim")
        failed_attempts = 0
        while failed_attempts < config.max_fsds_client_attempts:
            try:
                self.client = fsds.FSDSClient()
                self.client.confirmConnection()
                if controller:
                    self.client.enableApiControl(True)
                break
            except Exception:
                self.logger.info("Failed to connect client to sim on attempt %d" % failed_attempts)
                failed_attempts += 1
                time.sleep(1)
        if self.client is None:
            raise Exception("Failed to connect client")
        self.logger.info("Client connected to sim")

    def read_car_state(self):
        output = CarState()

        output.timestamp = time.time_ns()

        # Process car state:
        car_info = self.client.getCarState()
        state = self.client.simGetGroundTruthKinematics()
        output.car_pos = np.array([state.position.x_val, state.position.y_val, state.position.z_val])
        quats = np.array(
            [state.orientation.x_val, state.orientation.y_val, state.orientation.z_val, state.orientation.w_val]
        )
        output.car_hdg = np.asarray(Rotation.from_quat(quats).as_euler("yxz"))[2]
        output.world_linear_vel = np.array([state.linear_velocity.x_val, state.linear_velocity.y_val])

        # Low-pass filter the w since it is unusable due to noise when using high res camera:
        w = state.angular_velocity.z_val
        output.car_angular_vel = w

        output.car_angular_acc = state.angular_acceleration.z_val
        world_linear_acc = np.array([state.linear_acceleration.x_val, state.linear_acceleration.y_val])
        R = np.array([
            [np.cos(-output.car_hdg), -np.sin(-output.car_hdg)],
            [np.sin(-output.car_hdg), np.cos(-output.car_hdg)]
        ])
        output.car_linear_vel = R @ output.world_linear_vel
        output.car_linear_acc = R @ world_linear_acc
        output.car_speed = np.sqrt(np.square(output.car_linear_vel).sum())
        output.car_slip = np.arctan2(output.car_linear_vel[1], output.car_linear_vel[0])
        output.car_rpm = car_info.rpm

        output.car_steer_cmd = self.last_steer
        output.car_throttle = self.last_throttle
        output.car_d_steer = self.last_d_steer

        return output

    def read_referee_state(self):
        output = RefereeState()
        output.lap_time = None

        now = time.time_ns()
        referee_state = self.client.getRefereeState()
        output.collisions = referee_state.doo_counter
        output.lap_number = self.laps_driven

        if self.start_time is None:
            self.start_time = now
        output.time_passed_s = (now - self.start_time) / 1e9

        return output

    def set_controls(self, steer, throttle, d_steer):
        controls = fsds.CarControls(steering=-steer)
        if throttle >= 0:
            controls.throttle = throttle
        else:
            controls.brake = -throttle
        self.client.setCarControls(controls)
        self.last_steer = steer
        self.last_throttle = throttle
        self.last_d_steer = d_steer

    def get_gt_track(self):
        referee_state = self.client.getRefereeState()
        state = self.client.simGetGroundTruthKinematics()
        ref_pos = np.array([referee_state.initial_position.x, referee_state.initial_position.y, 0])
        car_pos = np.array([state.position.x_val, state.position.y_val, 0])
        quats = np.array(
            [state.orientation.x_val, state.orientation.y_val, state.orientation.z_val, state.orientation.w_val]
        )
        car_hdg = np.asarray(Rotation.from_quat(quats).as_euler("yxz"))[2]
        offset = car_pos - ref_pos

        blue_cones = []
        yellow_cones = []
        for cone in referee_state.cones:
            if cone['color'] == 1:
                blue_cones.append(np.array([cone['x'], cone['y'], 0]) + offset)
            elif cone['color'] == 0:
                yellow_cones.append(np.array([cone['x'], cone['y'], 0]) + offset)

        blue_cones = np.asarray(blue_cones)
        yellow_cones = np.asarray(yellow_cones)

        scaling_factor = 100
        axis_flips = [1, -1, 1]
        return car_pos, car_hdg, axis_flips * blue_cones / scaling_factor, axis_flips * yellow_cones / scaling_factor
