import logging
import sys
import time
import argparse
import config
from multiprocessing import Queue, Process, Event, Array, Lock
from queue import Empty
from multiprocessing.managers import SharedMemoryManager
from driving_system.driving_system import DrivingSystem
from utils.sim_helper import SimHelper
from utils.general import flush_queue, sm_array
from utils.data_storage import DataStorage
from utils.multiprocess_logging import log_receiver, configure_log_producer


def fill_data_storage(data_storage, queue):
    while True:
        try:
            data_storage.record_data(queue.get(block=False))
        except Empty:
            break


def runner_loop(sim_path, dynamics_types, maps, perception_data_queue, controller_data_queue, logging_queue, exit_event):
    logger = logging.getLogger("runner")
    configure_log_producer(logger, logging_queue)
    sim_helper = SimHelper(sim_path, logger)
    for dynamics_type in dynamics_types:
        data_storage = DataStorage(dynamics_type)
        for map_name in maps:
            logger.info("Starting map %s with dynamics type %s" % (map_name, dynamics_type))
            sim_helper.start_sim(map_name)

            odometry_data_queue = Queue()

            done_flag = Event()
            driver = DrivingSystem(dynamics_type, done_flag, odometry_data_queue, perception_data_queue, controller_data_queue, logging_queue)
            done_flag.wait()
            for queue in [perception_data_queue, controller_data_queue]:
                flush_queue(queue)
            fill_data_storage(data_storage, odometry_data_queue)
            #data_storage.save_dataset()
            del driver

            sim_helper.stop_sim()
            logger.info("Finished map %s with dynamics type %s" % (map_name, dynamics_type))

            if exit_event.is_set():
                return


def launch_debug_window(perception_data_queue, controller_data_queue, exit_event):
    app = QtWidgets.QApplication(sys.argv)
    vis = Visualizer(perception_data_queue, controller_data_queue, exit_event)
    vis.show()
    app.exec()
    if not exit_event.is_set():
        exit_event.set()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_path', type=str, default="/home/aleksi/Formula-Student-Driverless-Simulator/FSDS.sh")
    parser.add_argument('--dynamics_types', type=str, nargs="*", default=["dynamic_bicycle"])
    parser.add_argument('--maps', type=str, nargs="*", default=['TrainingMap', 'CompetitionMapTestday2', 'CompetitionMap1', 'CompetitionMap2'])
    parser.add_argument('--ci_mode', action="store_true")
    args = parser.parse_args()

    perception_data_queue = Queue()
    controller_data_queue = Queue()
    logging_queue = Queue()
    exit_event = Event()

    debug_process = None
    debug_queue = None
    if not args.ci_mode:
        from utils.debug_visualizer import Visualizer
        from PyQt6 import QtWidgets
        debug_process = Process(target=launch_debug_window, args=(perception_data_queue, controller_data_queue, exit_event))
        debug_process.start()

    log_receiver_process = Process(target=log_receiver, args=(logging_queue, ))
    log_receiver_process.start()
    runner_loop(args.sim_path, args.dynamics_types, args.maps, perception_data_queue, controller_data_queue, logging_queue, exit_event)
    logging_queue.put(None)
    log_receiver_process.join()

    if not args.ci_mode:
        debug_process.join()