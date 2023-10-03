import os
import config
import numpy as np
from scipy.signal import savgol_filter
from utils.maths import butter_lowpass_filter


class DataStorage:
    def __init__(self, dataset_name, source_dataset_name=None):
        self.current_data = []
        self.start_indices = []
        self.recorded_data = []
        self.initialized = False
        self.last_data = None

        self.dataset_dir = "dynamics_data/%s" % dataset_name
        os.makedirs(self.dataset_dir, exist_ok=True)

        if source_dataset_name is None:
            source_dataset_name = dataset_name
        source_dir = "dynamics_data/%s" % source_dataset_name
        try:
            self.start_indices = np.load("%s/start_indices.npy" % source_dir).tolist()
            self.recorded_data = np.load("%s/recorded_data.npy" % source_dir).tolist()
            print("Loaded dataset %s" % source_dir)
        except:
            print("FAILED TO LOAD INITIALIZATION DATASET %s" % source_dir)

    def reset(self):
        self.current_data = []
        self.start_indices = []
        self.recorded_data = []
        self.initialized = False
        self.last_data = None

    def record_data(self, odometry_data):
        world_vel = np.zeros_like(odometry_data[4:6])
        self.current_data.append(np.r_[odometry_data, world_vel])
        if not self.initialized:
            self.start_indices.append(len(self.recorded_data))
            self.initialized = True

    def save_dataset(self):
        self.initialized = False

        print(len(self.current_data))
        data_array = np.asarray(self.current_data)[1:-1]
        print(data_array.shape)
        dt = np.mean(data_array[1:, 0] - data_array[:-1, 0]) / 1e9

        print("Mean dt", 1e3*dt)
        print("Std dt", np.std(data_array[1:, 0] - data_array[:-1, 0]) / 1e6)

        if np.std(data_array[1:, 0] - data_array[:-1, 0]) / 1e6 > 1:
            print("HORRIBLE -----------------------------------------------------------------------------------------")

        result = np.c_[data_array[:, 0]]
        for j in range(1, data_array.shape[1]):
            # Calculate the angular acceleration with finite difference since the measurement is just white noise:
            if j == 9:
                data = data_array[:, 6]
                data = butter_lowpass_filter(data, 2, 1/dt, 2)
                data = 1e9*(data[1:] - data[:-1]) / (data_array[1:, 0] - data_array[:-1, 0])
                data = np.r_[data, 0]
            else:
                data = data_array[:, j]
            result = np.c_[result, data]

        self.recorded_data.extend(result.tolist())
        self.current_data = []
        self.last_data = result.astype(np.float64)
        print("LAST DATA", self.last_data.shape, self.last_data.dtype)

        np.save("%s/start_indices" % self.dataset_dir, np.asarray(self.start_indices))
        np.save("%s/recorded_data" % self.dataset_dir, np.asarray(self.recorded_data))

    # Resample at constant dt and savgol filter the data:
    def get_training_data(self):
        n_inputs = 7
        data_idx = [
            # Inputs (0-6]:
            (3, False),   # HDG
            (4, False),   # VX
            (5, False),   # VY
            (6, False),   # W
            (10, False),  # STEER
            (11, False),  # THROTTLE
            (12, False),  # DSTEER
            # Outputs (7-12):
            (13, False),  # VX WORLD
            (14, False),  # VY WORLD
            (6, False),   # W
            (7, False),   # AX
            (8, False),   # AY
            (6, True),    # dW
        ]

        original = []
        filtered = []
        start_indices_array = np.asarray(self.start_indices)
        data_array = np.asarray(self.recorded_data)

        for i in range(len(start_indices_array)):
            start = start_indices_array[i]
            tmp_original = None
            tmp_filtered = None
            if i == len(start_indices_array) - 1:
                end = len(data_array)
            else:
                end = start_indices_array[i + 1]

            dt = np.mean(data_array[start+1:end, 0] - data_array[start:end-1, 0]) / 1e9
            window_size = int(config.p / dt)
            if end - start >= window_size:
                for idx, deriv in data_idx:
                    selected = data_array[start:end, idx]

                    if idx in [6, 7, 8]:
                        selected = butter_lowpass_filter(selected, 2, 1 / dt, order=2)

                    if deriv:
                        savgol = savgol_filter(
                            selected, window_length=window_size, polyorder=config.savgol_d_k, deriv=1
                        )[:-1] / dt
                        # For dw, use finite difference derivative as reference:
                        if idx == 6:
                            selected = data_array[start:end, 9]
                    else:
                        # Filter all but hdg or controls:
                        if idx not in [3, 10, 11, 12]:
                            savgol = savgol_filter(
                                selected, window_length=window_size, polyorder=config.savgol_k, deriv=0
                            )[:-1]
                        else:
                            savgol = selected[:-1]

                    if tmp_original is None:
                        tmp_original = selected[:-1]
                        tmp_filtered = savgol
                    else:
                        tmp_original = np.c_[tmp_original, selected[:-1]]
                        tmp_filtered = np.c_[tmp_filtered, savgol]

                original.extend(tmp_original)
                filtered.extend(tmp_filtered)

        original = np.asarray(original)
        filtered = np.asarray(filtered)

        if len(original) == 0 or len(filtered) == 0:
            original = np.zeros([1, len(data_idx)])
            filtered = np.zeros([1, len(data_idx)])

        return original[:-1, :n_inputs], original[1:, n_inputs:], filtered[:-1, :n_inputs], filtered[1:, n_inputs:]