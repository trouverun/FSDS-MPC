import logging
import time
import config
import fsds
import numpy as np
from scipy.spatial.transform import Rotation
from multiprocessing.shared_memory import SharedMemory
from driving_system.control_system.controller import Controller
from driving_system.queue_messages import ControllerData
from driving_system.control_system.planning import find_centerline
from utils.multiprocess_logging import configure_log_producer
from utils.general import sm_array


def control_loop(dynamics_type, ready_barrier,
                 car_state_memory_name, car_state_lock,
                 cone_map_memory_name, num_cones, cone_map_lock,
                 shared_controls, data_queue, logging_queue,
                 done_flag):
    logger = logging.getLogger("driving_system")
    configure_log_producer(logger, logging_queue)

    client = fsds.FSDSClient()
    client.confirmConnection()
    controller = Controller(dynamics_type, logger)
    ready_barrier.wait()
    time.sleep(1)

    car_state_memory = SharedMemory(car_state_memory_name)
    shared_car_state = sm_array(car_state_memory, config.car_shared_state_size)
    car_state_lock = car_state_lock

    cone_map_memory = SharedMemory(cone_map_memory_name)
    shared_cone_map = sm_array(cone_map_memory, (2, config.n_map_cones, 6))

    stuck_steps = 0
    num_measured_delays = 1
    delays = np.zeros([100])
    failed_solves = 0
    laps_done = 0

    referee_state = client.getRefereeState()
    state = client.simGetGroundTruthKinematics()
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
    gt_blue_cones = axis_flips * blue_cones / scaling_factor
    gt_yellow_cones = axis_flips * yellow_cones / scaling_factor

    max_speed = config.car_initial_max_speed
    while True:
        t1 = time.time_ns()
        if done_flag.is_set():
            car_state_memory.close()
            cone_map_memory.close()
            return

        output = ControllerData()

        car_state_lock.acquire()
        car_state = shared_car_state.copy()
        car_state_lock.release()

        if car_state[3] < 1e-0:
            stuck_steps += 1
        else:
            stuck_steps = 0

        # Sync the cone map during first lap
        # (after that freeze it so that mpc optimal solutions don't change due to map updates = less fragile solver)
        cone_map_lock.acquire()
        num_blue_cones = num_cones[0]
        num_yellow_cones = num_cones[1]
        cone_map = shared_cone_map.copy()
        cone_map_lock.release()
        blue_cones = cone_map[0, :num_blue_cones]
        yellow_cones = cone_map[1, :num_yellow_cones]

        # Find the reference track centerline and get control with MPC:
        solver_start = time.time_ns()
        midpoints = find_centerline(
            car_state[:2], car_state[2],
            blue_cones.copy(), yellow_cones.copy(), no_path_limit=True
        )
        # Delay compensate based on a moving median of recent measured delays:
        delay_s = np.median(delays[(len(delays)-num_measured_delays):])
        controller_out = controller.get_control(car_state, midpoints, max_speed, delay_s)
        solver_stop = time.time_ns()

        shared_controls[0] = controller_out.steer
        shared_controls[1] = controller_out.throttle
        shared_controls[2] = controller_out.d_steer

        solver_delay_s = (solver_stop-solver_start)/1e9
        delays[:-1] = delays[1:]
        delays[-1] = solver_delay_s
        num_measured_delays = max(num_measured_delays+1, 100)

        referee_state = client.getRefereeState()
        if len(referee_state.laps) > laps_done:
            # max_speed += config.lap_speed_increase
            max_speed = config.car_initial_max_speed + config.lap_speed_increase
            laps_done += 1

        if not controller_out.solver_success:
            failed_solves += 1
            logger.warning("Fail")
        else:
            failed_solves = 0

        if failed_solves > config.max_failed_mpc_solves or referee_state.doo_counter > config.max_collisions or stuck_steps > config.max_stuck_steps or controller_out.stop:
            done_flag.set()

        # Fill output object:
        output.car_state = car_state
        output.blue_cones = blue_cones
        output.yellow_cones = yellow_cones
        output.midpoints = midpoints
        output.trajectory = controller_out.trajectory
        output.solver_time_ms = 1e3*solver_delay_s
        data_queue.put(output)

        t2 = time.time_ns()
        t_passed_s = (t2-t1)/1e9
        time.sleep(max(0, config.mpc_dt-t_passed_s))