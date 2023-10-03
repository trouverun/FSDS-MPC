import logging
import time
import pymap3d
import config
import numpy as np
import torch
import fsds
from scipy import stats
from threading import Thread, Lock
from driving_system.control_system.torch_dynamics.dynamic_bicycle import DynamicBicycle
from scipy.spatial.transform import Rotation
from multiprocessing.shared_memory import SharedMemory
from utils.general import sm_array


class StateEstimator:
    def __init__(self, ready_barrier, car_state_memory_name, car_state_lock,
                 shared_controls, odometry_queue, logging_queue, done_flag, collect_raw_odometry_data=False):
        self.car_state_memory = SharedMemory(car_state_memory_name)
        self.shared_car_state = sm_array(self.car_state_memory, config.car_shared_state_size)
        self.car_state_lock = car_state_lock
        self.shared_controls = shared_controls
        self.odometry_queue = odometry_queue
        self.logging_queue = logging_queue
        self.done_flag = done_flag
        self.collect_raw_odometry_data = collect_raw_odometry_data

        self.dynamics = DynamicBicycle()
        self.state_mu = torch.zeros(9)
        self.state_sigma = torch.zeros([9, 9])
        self.client = fsds.FSDSClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        home_geo = self.client.getGpsData(gps_name='Gps', vehicle_name='FSCar')
        self.lat_long_0 = np.array([home_geo.gnss.geo_point.latitude, home_geo.gnss.geo_point.longitude, home_geo.gnss.geo_point.altitude])
        self.iterations_since_imu = 0
        self.iterations_since_gps = 0

        ready_barrier.wait()

        # Dynamics covariance:
        self.R = torch.diag(torch.tensor([
            1,                 # x
            1,                 # y
            np.deg2rad(1),     # hdg
            1,                 # vx
            1,                 # vy
            np.deg2rad(0.25),   # w
            2.5,              # ax
            2.5,              # ay
            np.deg2rad(10)     # dw
        ], dtype=torch.float32))

        # Measurement covariance:
        self.Q = torch.diag(torch.tensor([
            0.25,              # GPS
            0.25,              # GPS
            np.deg2rad(1),  # IMU heading
            0.1,              # GSS
            0.1,              # GSS
            np.deg2rad(2.5),    # IMU w
            1.25,             # IMU acc
            1.25,             # IMU ACC
            0                 # IMU dww
        ], dtype=torch.float32))

        self.controls = torch.zeros(3)

    def _discrete_state_transition(self, state, controls):
        input_tensor = torch.hstack([state[2:6], controls]).unsqueeze(0)
        model_out = self.dynamics(input_tensor)[0]
        x_out = torch.hstack([model_out, torch.zeros([3])])
        dx_out = torch.hstack([torch.zeros([3]), model_out])
        x_select = torch.hstack([torch.ones(6), torch.zeros(3)])
        dx_select = torch.ones_like(x_select) - x_select
        return x_select * state + config.state_estimator_dt * x_out + dx_select * dx_out

    def sensor_loop(self):
        while True:
            t1 = time.time_ns()

            if self.done_flag.is_set():
                self.car_state_memory.close()
                return

            self.controls[0] = self.shared_controls[0]
            self.controls[1] = self.shared_controls[1]
            self.controls[2] = self.shared_controls[2]

            steer = self.controls[0].item()
            throttle = self.controls[1].item()
            fsds_controls = fsds.CarControls(steering=-steer)
            if throttle >= 0:
                fsds_controls.throttle = throttle
            else:
                fsds_controls.brake = -throttle
            self.client.setCarControls(fsds_controls)

            selector = torch.zeros(9)
            H = torch.eye(9)
            car_pos = torch.zeros(2)
            w = 0
            car_acceleration = torch.zeros(2)

            timestamp = time.time_ns()

            # hdg from IMU:
            imu = self.client.getImuData(imu_name='Imu', vehicle_name='FSCar')
            quats = np.array(
                [imu.orientation.x_val, imu.orientation.y_val, imu.orientation.z_val, imu.orientation.w_val])
            car_hdg = np.asarray(Rotation.from_quat(quats).as_euler("yxz"))[2]
            selector[2] = 1

            # GSS velocity reading:
            gss = self.client.getGroundSpeedSensorData(vehicle_name='FSCar')
            car_vel = np.array([gss.linear_velocity.x_val, gss.linear_velocity.y_val])
            R = np.array([
                [np.cos(-car_hdg), -np.sin(-car_hdg)],
                [np.sin(-car_hdg), np.cos(-car_hdg)]
            ])
            car_vel = R @ car_vel
            selector[3:5] = 1

            # Linear acceleration and w from IMU (lower hz since the simulator cannot update the internal state fast enough):
            if self.iterations_since_imu == 2:
                car_acceleration = np.array([imu.linear_acceleration.x_val, imu.linear_acceleration.y_val, imu.linear_acceleration.z_val])
                w = np.array([imu.angular_velocity.x_val, imu.angular_velocity.y_val, imu.angular_velocity.z_val])[2]
                quats = np.array(
                    [imu.orientation.x_val, imu.orientation.y_val, imu.orientation.z_val, imu.orientation.w_val])
                car_hdg = np.asarray(Rotation.from_quat(quats).as_euler("yxz"))[2]
                selector[5:8] = 1
                self.iterations_since_imu = 0
            else:
                self.iterations_since_imu += 1

            # GPS position reading at 10Hz:
            if self.iterations_since_gps == 5:
                gps = self.client.getGpsData(gps_name='Gps', vehicle_name='FSCar')
                lat_long = np.array(
                    [gps.gnss.geo_point.latitude, gps.gnss.geo_point.longitude, gps.gnss.geo_point.altitude])
                car_pos = pymap3d.geodetic2enu(
                    lat_long[0], lat_long[1], lat_long[2],
                    self.lat_long_0[0], self.lat_long_0[1], self.lat_long_0[2]
                )
                car_pos = torch.tensor([car_pos[0], car_pos[1], 0])
                selector[0:2] = 1
                self.iterations_since_gps = 0
            else:
                self.iterations_since_gps += 1

            G = torch.func.jacrev(self._discrete_state_transition, argnums=(0))(self.state_mu, self.controls)
            predicted_mu = self._discrete_state_transition(self.state_mu, self.controls)
            predicted_sigma = G @ self.state_sigma @ G.T + self.R

            if not self.collect_raw_odometry_data:
                if selector[5]:
                    ci = stats.norm.interval(0.9, loc=predicted_mu[5].detach().item(), scale=np.sqrt(predicted_sigma[5, 5].detach().item()))
                    if not ci[0] < w < ci[1]:
                        selector[5] = 0
                        print("unlikely measurement: %f < %f < %f" % (ci[0], w, ci[1]))

            selector = torch.diag(selector)
            measurement = selector @ torch.tensor([
                car_pos[0],
                car_pos[1],
                car_hdg,
                car_vel[0],
                car_vel[1],
                w,
                car_acceleration[0],
                car_acceleration[1],
                0
            ], dtype=torch.float32)
            H = selector @ H

            # Fix huge differences in angles due to (angle+n*2pi - angle):
            residual = measurement - H @ predicted_mu
            residual[2] = torch.atan2(torch.sin(measurement[2]-predicted_mu[2].detach()), torch.cos(measurement[2]-predicted_mu[2].detach()))

            kalman_gain = predicted_sigma @ H.T @ torch.pinverse(H @ predicted_sigma @ H.T + self.Q)
            self.state_mu = predicted_mu + kalman_gain @ residual
            self.state_sigma = (torch.eye(9) - kalman_gain @ H) @ predicted_sigma

            car_state = np.r_[
                self.state_mu.detach().numpy(),
                self.controls.numpy()
            ]

            self.car_state_lock.acquire()
            self.shared_car_state[:] = car_state
            self.car_state_lock.release()

            # When learning initial model we have nothing useful to kalman filter with,
            # so use raw measurements when available (lower Hz):
            if self.collect_raw_odometry_data:
                if torch.all(torch.diagonal(selector)[5:8]):
                    self.odometry_queue.put(np.r_[timestamp, measurement, self.controls.numpy()])
            else:
                self.odometry_queue.put(np.r_[timestamp, measurement, self.controls.numpy()])
                # self.odometry_queue.put(np.r_[timestamp, car_state])

            t2 = time.time_ns()
            t_passed_s = (t2-t1)/1e9
            sleep_duration = max(0, config.state_estimator_dt-t_passed_s)
            time.sleep(sleep_duration)


def launch_state_estimator(ready_barrier, car_state_memory_name, car_state_lock,
                           shared_controls, odometry_queue, logging_queue, done_flag, collect_raw_odometry_data):
    estimator = StateEstimator(ready_barrier, car_state_memory_name, car_state_lock,
                               shared_controls, odometry_queue, logging_queue, done_flag, collect_raw_odometry_data)
    estimator.sensor_loop()
    print("STATE ESTIMATOR IS DED")