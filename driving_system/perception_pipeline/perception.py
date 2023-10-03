import logging
import sys
import numpy as np
import torch
import time
import cv2
import torch.backends.cudnn as cudnn
import config
import fsds
from torchvision.transforms.functional import resize
from multiprocessing.shared_memory import SharedMemory
from scipy.spatial.transform import Rotation
from driving_system.queue_messages import PerceptionData
from driving_system.perception_pipeline.keypoints.kp_model import KeypointModel
from driving_system.perception_pipeline.yolov7.models.yolo import Model
from driving_system.perception_pipeline.yolov7.utils.general import non_max_suppression
from utils.multiprocess_logging import configure_log_producer
from utils.general import sm_array
from utils.maths import rp_to_trans, trans_inv, is_point_in_triangle


def perception_loop(ready_barrier, car_state_memory_name, car_state_lock,
                    cone_map_memory_name, num_cones, cone_map_lock,
                    data_queue, logging_queue, done_flag):
    logger = logging.getLogger("perception")
    configure_log_producer(logger, logging_queue)

    logger.info("loading pytorch networks")
    sys.path.append('driving_system/perception_pipeline/yolov7')
    yolo_model = Model(cfg='driving_system/perception_pipeline/yolov7/yolov7-tiny.yaml', nc=10).to('cuda:0')
    yolo_model.load_state_dict(torch.load('driving_system/perception_pipeline/yolov7/yolo_model.pt'))
    yolo_model.eval()
    yolo_model.half()
    sys.path.append('driving_system/perception_pipeline/keypoints/')
    kp_model = KeypointModel().to('cuda:0')
    kp_model.load_state_dict(torch.load('driving_system/perception_pipeline/keypoints/kp_weights.pt'))
    kp_model.eval()
    kp_model.half()
    cudnn.benchmark = False
    client = fsds.FSDSClient()
    client.confirmConnection()
    ready_barrier.wait()

    car_state_memory = SharedMemory(car_state_memory_name)
    shared_car_state = sm_array(car_state_memory, config.car_shared_state_size)
    car_state_lock = car_state_lock

    cone_map_memory = SharedMemory(cone_map_memory_name)
    shared_cone_map = sm_array(cone_map_memory, (2, config.n_map_cones, 6))
    blue_cones = []
    yellow_cones = []
    while True:
        if done_flag.is_set():
            car_state_memory.close()
            cone_map_memory.close()
            return

        output = PerceptionData()

        car_state_lock.acquire()
        car_state = shared_car_state.copy()
        car_state_lock.release()
        car_pos = np.r_[car_state[:2], 0]
        car_hdg = car_state[2]

        t1 = time.time_ns()
        images = client.simGetImages(
            [fsds.ImageRequest(camera_name=config.camera_name, image_type=fsds.ImageType.Scene, pixels_as_float=False, compress=False)],
            vehicle_name='FSCar'
        )
        byte_data = images[0].image_data_uint8
        camera_frame = fsds.string_to_uint8_array(byte_data).reshape([config.camera_resolution, config.camera_resolution, 3])[:, :, ::-1].astype(np.uint8)

        # client.getLidarData(lidar_name='Lidar', vehicle_name='FSCar')
        t2 = time.time_ns()
        output.camera_ms = (t2-t1) / 1e6

        # Find bounding boxes for all cones in frame
        with torch.no_grad():
            torch_image = torch.from_numpy(camera_frame).unsqueeze(0).permute(0, 3, 1, 2).half().to('cuda:0') / 255.0
            t1 = time.time_ns()
            bbox_pred = yolo_model(torch_image)[0]
            bbox_pred = non_max_suppression(bbox_pred, conf_thres=config.min_bbox_conf)[0].cpu()
            t2 = time.time_ns()
            output.yolo_ms = (t2-t1) / 1e6

            # Crop and find keypoints on the detected cones:
            t1 = time.time_ns()
            torch_image = torch_image.squeeze(0)
            ratios = []
            crops = torch.zeros([len(bbox_pred), 3, config.kp_resize_h, config.kp_resize_w]).half().to('cuda:0')
            new_i = 0
            indices = []
            for i, det in enumerate(bbox_pred):
                det = det.unsqueeze(0)
                if len(det):
                    for *xyxy, conf, cone_class in reversed(det):
                        if conf > config.min_bbox_conf and cone_class in [config.BLUE_CONE, config.YELLOW_CONE]:
                            top = xyxy[1].to(torch.int32)
                            bot = xyxy[3].to(torch.int32)
                            left = torch.max(torch.zeros_like(xyxy[0]), xyxy[0].to(torch.int32)).to(torch.int32)
                            right = xyxy[2].to(torch.int32)
                            height = bot - top
                            width = right - left
                            if height > config.kp_min_h and width > config.kp_min_w and height / width < 5 / 3:
                                crops[new_i] = resize(torch_image[:, top:bot, left:right], [config.kp_resize_h, config.kp_resize_w])
                                ratios.append(torch.tensor([config.kp_resize_w / width, config.kp_resize_h / height]))
                                indices.append(i)
                                new_i += 1
            points = kp_model(crops[:new_i]).cpu()
            t2 = time.time_ns()
            output.kp_ms = (t2-t1) / 1e6

        # Get transformation matrix from camera to cone
        t1 = time.time_ns()
        R_car = Rotation.from_euler("xyz", [0, 0, car_hdg-np.deg2rad(90)]).as_matrix()
        P_car = car_pos
        cones = []
        for i in range(len(points)):
            cones.append(pnp_process_cones(points[i], ratios[i], bbox_pred[indices][i], R_car, P_car))
        cones = np.asarray(cones)
        t2 = time.time_ns()
        output.pnp_ms = (t2-t1) / 1e6

        try:
            observed_blue_cones = cones[cones[:, 0] == config.BLUE_CONE][:, 1:4]
            blue_pixels = cones[cones[:, 0] == config.BLUE_CONE][:, 4:]
        except Exception:
            observed_blue_cones = np.zeros([0, 3])
            blue_pixels = np.array([])
        try:
            observed_yellow_cones = cones[cones[:, 0] == config.YELLOW_CONE][:, 1:4]
            yellow_pixels = cones[cones[:, 0] == config.YELLOW_CONE][:, 4:]
        except Exception:
            observed_yellow_cones = np.zeros([0, 3])
            yellow_pixels = np.array([])

        # Define a field of vision triangle in front of the car:
        R_car = Rotation.from_euler("xyz", [0, 0, -(car_hdg - np.deg2rad(90))]).as_matrix()
        v1 = np.array([0, 0])
        v2 = v1 + [
            -np.sin(np.deg2rad(config.mapping_vision_adjuster * config.camera_fov / 2)) * config.max_valid_cone_distance,
            np.cos(np.deg2rad(config.mapping_vision_adjuster * config.camera_fov / 2)) * config.mapping_vision_adjuster * config.max_valid_cone_distance
        ]
        v3 = v1 + [
            -np.sin(-np.deg2rad(config.mapping_vision_adjuster * config.camera_fov / 2)) * config.max_valid_cone_distance,
            np.cos(-np.deg2rad(config.mapping_vision_adjuster * config.camera_fov / 2)) * config.mapping_vision_adjuster * config.max_valid_cone_distance
        ]

        # Update the cone map with new observations
        t1 = time.time_ns()
        if len(observed_blue_cones):
            update_cones(blue_cones, observed_blue_cones, R_car, car_pos, v1, v2, v3)
        if len(observed_yellow_cones):
            update_cones(yellow_cones, observed_yellow_cones, R_car, car_pos, v1, v2, v3)
        t2 = time.time_ns()
        output.map_update_ms = (t2-t1) / 1e6

        # Sync the shared memory cone map
        if len(blue_cones):
            blue_cones_array = np.asarray(blue_cones)
        else:
            blue_cones_array = np.zeros([0, 6])
        blue_cones_array = np.r_[blue_cones_array, np.zeros([config.n_map_cones-len(blue_cones), 6])]
        if len(yellow_cones):
            yellow_cones_array = np.asarray(yellow_cones)
        else:
            yellow_cones_array = np.zeros([0, 6])
        yellow_cones_array = np.r_[yellow_cones_array, np.zeros([config.n_map_cones-len(yellow_cones), 6])]
        cone_map_lock.acquire()
        shared_cone_map[0, :] = blue_cones_array
        shared_cone_map[1, :] = yellow_cones_array
        num_cones[0] = len(blue_cones)
        num_cones[1] = len(yellow_cones)
        cone_map_lock.release()

        output.camera_frame = camera_frame
        output.blue_pixels = blue_pixels
        output.yellow_pixels = yellow_pixels
        data_queue.put(output)


def pnp_process_cones(img_points, ratios, det, R_car, P_car):
    left = det[0].to(torch.int32).numpy()
    top = det[1].to(torch.int32).numpy()
    right = det[2].to(torch.int32).numpy()
    bot = det[3].to(torch.int32).numpy()
    cone_class = det[5].to(torch.int32).numpy()

    img_points = (img_points.reshape(7, 2) / ratios).numpy() + [left, top]
    retval, rvec, tvec = cv2.solvePnP(config.cone_3d_points, img_points, config.K, np.zeros(4))
    if retval:
        T = rp_to_trans(np.eye(3), tvec)
        T = config.camera_T @ trans_inv(T)
        cone_pos = T @ np.array([0, 0, 0, 1]) * np.array([-1, -1, 0, 0])
        if config.min_valid_cone_distance < cone_pos[1] < config.max_valid_cone_distance:
            cone_class = np.array([cone_class])
            return np.concatenate([
                cone_class, (R_car @ cone_pos[:3] + P_car[:3]), np.array([left, top, right, bot]), img_points.flatten()
            ])

    return np.zeros(22)


def update_cones(cones, measured_cones, R_car, car_pos, v1, v2, v3):
    n_initial_cones = len(cones)
    cone_array = np.asarray(cones)
    observed_cones = []

    # Adjust cone variance based on distance
    cone_distances = np.sqrt(np.square(car_pos[:2] - measured_cones[:, :2]).sum(axis=1))
    adjusted = (cone_distances - config.variance_increase_distance) / (config.variance_increase_distance / config.additional_cone_pos_variance)
    dist_variance = np.where(cone_distances > 10, adjusted, np.zeros_like(cone_distances))
    dist_variance = np.clip(dist_variance, 0, config.additional_cone_pos_variance)
    measurement_variance = dist_variance + config.cone_position_variance

    # For each measured cone, calculated the likelihood that it corresponds to an existing cone in the map.
    # If likelihood is below a threshold, a new cone is added to the map,
    # otherwise the position of the most likely match is adjusted
    for i, measured_cone_pos in enumerate(measured_cones):
        pz = np.zeros(n_initial_cones + 1)
        # Threshold probability for creating a new cone:
        pz[n_initial_cones] = config.cone_ekf_alpha
        if n_initial_cones:
            cone_pos_mean = cone_array[:, :3]
            cone_pos_variance = cone_array[:, 3]
            dist = np.sqrt(np.square(measured_cone_pos - cone_pos_mean).sum(axis=1))
            dist = np.minimum(dist, 10*np.ones_like(dist))   # Prevent overflow from exp(>(10*10))
            adjusted_variance = measurement_variance[i] + cone_pos_variance
            # Correspondence likelihood for all existing cones in the map (adapted from FastSLAM):
            pz[:n_initial_cones] = (
                    np.power(np.sqrt(np.abs(2 * np.pi * adjusted_variance)), -0.5)
                    * np.exp(-0.5 * dist * (1 / adjusted_variance) * dist)
            )
        # Find the correspondence index j:
        pz = pz / pz.sum()
        j = np.argmax(pz)
        if j != n_initial_cones:
            # Matched an existing cone, adjust the position estimate with the measurement and measurement variance:
            K = cone_pos_variance[j] * 1 / measurement_variance[i]
            cones[j][:3] += K * (measured_cone_pos - cone_array[j, :3])
            cones[j][3] *= 1 - K
            observed_cones.append(j)
        else:
            # Create a new cone:
            j = len(cones)
            cones.append(np.zeros(6))
            cones[j][:3] = measured_cone_pos
            cones[j][3] = measurement_variance[i]
        cones[j][4] = cones[j][4] + 1

    # For all existing cones which are inside the field of vision triangle, but weren't matched to any measured cones,
    # decrease the likelihood of a correct estimate, and remove if below a threshold:
    all_previous_cone_i = [*range(0, n_initial_cones)]
    not_observed_previous_cone_i = sorted(np.setdiff1d(all_previous_cone_i, observed_cones).tolist())
    not_observed_previous_cone_i.reverse()
    if len(not_observed_previous_cone_i):
        rotated_cones = (R_car @ (cone_array[:, :3] - car_pos[:3]).T).T
        for cone_i in not_observed_previous_cone_i:
            if np.sqrt(np.square(rotated_cones[cone_i, :2]).sum()) > 2:
                if is_point_in_triangle(rotated_cones[cone_i, :2], v1, v2, v3):
                    cones[cone_i][5] += 1
                    if cones[cone_i][4] / cones[cone_i][5] < config.delete_threshold:
                        cones.pop(cone_i)