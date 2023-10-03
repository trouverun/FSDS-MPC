
import config
from multiprocessing import Queue, Process, Event, Barrier, Array, Lock
from multiprocessing.managers import SharedMemoryManager
from driving_system.perception_pipeline.perception import perception_loop
from driving_system.state_estimation.state_estimation import launch_state_estimator
from driving_system.control_system.control_system import control_loop
from utils.general import sm_array


class DrivingSystem:
    def __init__(self, dynamics_type, done_flag, odometry_data_queue, perception_data_queue, controller_data_queue, logging_queue):
        self.done_flag = done_flag
        self.smm = SharedMemoryManager()
        self.smm.start()

        n_bytes = config.car_shared_state_size * 4
        self.car_state_memory = self.smm.SharedMemory(size=n_bytes)
        self.shared_car_state = sm_array(self.car_state_memory, config.car_shared_state_size)
        self.car_state_lock = Lock()

        self.controls = Array('f', [0.0, 0.0, 0.0], lock=True)

        n_bytes = 2 * config.n_map_cones * 6 * 4
        self.cone_map_memory = self.smm.SharedMemory(size=n_bytes)
        self.shared_cone_map = sm_array(self.cone_map_memory, (2, config.n_map_cones, 6))
        self.num_cones = Array('i', [0, 0], lock=False)
        self.cone_map_lock = Lock()

        n_processes = 3
        if dynamics_type == "manual_control":
            n_processes = 2
        self.all_ready_barrier = Barrier(n_processes)

        raw_odometry = dynamics_type == "manual_control"

        self.subsystem_processes = [
            Process(target=launch_state_estimator, args=(
                self.all_ready_barrier, self.car_state_memory.name, self.car_state_lock,
                self.controls, odometry_data_queue, logging_queue, done_flag, raw_odometry)),
            Process(target=perception_loop, args=(
                self.all_ready_barrier, self.car_state_memory.name, self.car_state_lock,
                self.cone_map_memory.name, self.num_cones, self.cone_map_lock,
                perception_data_queue, logging_queue, done_flag
            )),
            Process(target=control_loop, args=(
                dynamics_type, self.all_ready_barrier,
                self.car_state_memory.name, self.car_state_lock,
                self.cone_map_memory.name, self.num_cones, self.cone_map_lock,
                self.controls, controller_data_queue, logging_queue,
                done_flag
            ))
        ]

        # Manual data collection done using sim rendering, no reason to use cameras and degrade sensor readings:
        if dynamics_type == "manual_control":
            self.subsystem_processes.pop(1)

        for process in self.subsystem_processes:
            process.start()

    def __del__(self):
        for process in self.subsystem_processes:
            process.join()

        self.smm.shutdown()