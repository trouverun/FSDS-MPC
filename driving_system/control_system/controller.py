import time
import config
import numpy as np
from driving_system.control_system.solver import Solver
from driving_system.control_system.torch_dynamics.dynamic_bicycle import DynamicBicycle
from driving_system.queue_messages import ControllerOutput
from utils.game_controller import XboxController


class Controller:
    def __init__(self, model_type, logger):
        self.model_type = model_type
        if model_type != "manual_control":
            self.torch_model = DynamicBicycle()
            self.solver = Solver(model_type)
        else:
            self.game_controller = XboxController()
        self.logger = logger
        self.prediction_horizon = None
        self.time_passed_s = 0

    def _extract_trajectory(self, state):
        return np.c_[
            state[:, :2],
            state[:, 3],
        ]

    def get_control(self, car_state, midpoints, max_speed, delay_comp_s):
        output = ControllerOutput()
        output.timestamp = time.time_ns()

        state = np.r_[car_state[:6], car_state[9:11]]

        if self.model_type != "manual_control":
            drivetrain_params, bicycle_params = self.torch_model.extract_params()
            state = self.solver.delay_compensation(state, car_state[11], drivetrain_params, bicycle_params, delay_comp_s)

        state[3] = np.clip(state[3], 0, np.inf)

        predicted_states = np.repeat(state, config.mpc_horizon + 1).reshape([len(state), config.mpc_horizon + 1]).T
        predicted_states = np.c_[predicted_states, np.zeros(config.mpc_horizon+1)]

        if self.model_type == "manual_control":
            d_steer = 0
            steer, accel, decel, stop, steer_sine, accel_sine = self.game_controller.read()
            steer = np.clip(steer, -config.steer_value_max, config.steer_value_max)
            choices = [accel, -decel]
            throttle = choices[np.argmax(np.abs(choices))]
            throttle = np.clip(throttle, -config.throttle_value_max, config.throttle_value_max)

            if accel_sine:
                throttle = config.throttle_value_max/2 + config.throttle_value_max/2 * np.sin((self.time_passed_s / 7.5 * 2 * np.pi) + np.arcsin(-1))
                self.time_passed_s += config.mpc_dt
            elif steer_sine:
                throttle = 0.075
                steer = config.steer_value_max * np.sin(self.time_passed_s / 7.5 * 2 * np.pi)
                d_steer = config.steer_value_max * 2 * np.pi / 7.5 * np.cos(self.time_passed_s / 7.5 * 2 * np.pi)
                self.time_passed_s += config.mpc_dt
            else:
                self.time_passed_s = 0

            solver_success = True
            output.stop = stop
        else:
            if len(midpoints) < 2:
                output.predicted_trajectory = self._extract_trajectory(predicted_states)
                solver_success = False
                if self.prediction_horizon is not None:
                    self.prediction_horizon[:-1] = self.prediction_horizon[1:]
                    x = self.prediction_horizon[0]
                    steer = x[6]
                    throttle = x[7]
                else:
                    steer = 0
                    throttle = 0
                d_steer = 0
            else:
                dt = np.linspace(1/20, 1/10, self.solver.N+1)
                # dt = np.repeat(config.mpc_dt, self.solver.N+1)
                params = np.r_[drivetrain_params, bicycle_params]
                self.solver.initialize(state, midpoints, max_speed, dt, params)
                try:
                    steer, throttle, predicted_states, d_steer = self.solver.solve()
                    solver_success = True
                    self.prediction_horizon = predicted_states
                except Exception as e:
                    solver_success = False
                    if self.prediction_horizon is not None:
                        self.prediction_horizon[:-1] = self.prediction_horizon[1:]
                        x = self.prediction_horizon[0]
                        steer = x[6]
                        throttle = x[7]
                    else:
                        steer = 0
                        throttle = 0
                    d_steer = 0

        output.trajectory = self._extract_trajectory(predicted_states)
        output.steer = steer
        output.throttle = throttle
        output.d_steer = d_steer
        output.solver_success = solver_success

        return output