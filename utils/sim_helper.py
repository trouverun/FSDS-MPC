import os
import signal
import subprocess
import time


class SimHelper:
    def __init__(self, sim_executable_path, logger):
        self.executable_path = sim_executable_path
        self.logger = logger
        self.simulator_pid = None
        self.simulator_launcher_process = None

    def start_sim(self, map_name):
        try:
            self.simulator_pid = int(subprocess.check_output(['pidof', "-s", 'Blocks-Linux-Test']))
            self.stop_sim(wait=True)
        except:
            pass

        self.logger.info("Starting simulator")
        try:
            self.simulator_launcher_process = subprocess.Popen([self.executable_path, map_name], shell=False)
            time.sleep(10)
            self.simulator_pid = int(subprocess.check_output(['pidof', "-s", 'Blocks-Linux-Test']))
        except Exception:
            raise Exception("Failed to start simulator")
        self.logger.info("Simulator start ok")

    def _init_state(self):
        self.simulator_pid = None
        self.simulator_launcher_process = None

    def stop_sim(self, wait=True):
        os.kill(self.simulator_pid, signal.SIGKILL)
        time.sleep(5)
        if self.simulator_launcher_process is not None:
            self.simulator_launcher_process.kill()
            self.simulator_launcher_process.wait()
        if wait:
            time.sleep(5)