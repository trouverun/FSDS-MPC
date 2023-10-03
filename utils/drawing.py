import pygame
import casadi
import numpy as np
import matplotlib as plt
import config
import cv2
from scipy import interpolate
from utils.maths import rotate_around


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
DEBUG = (0, 255, 255)


def draw_detections(camera_frame, blue_detections, yellow_detections):
    camera_frame = cv2.resize(
        camera_frame, dsize=(int(config.camera_resolution/3), int(config.camera_resolution/3)),
        interpolation=cv2.INTER_LANCZOS4
    )
    surf = pygame.surfarray.make_surface(camera_frame.swapaxes(0, 1))
    for detections, color in [(blue_detections, BLUE), (yellow_detections, YELLOW)]:
        for det in detections:
            det /= 3
            det = det.astype(np.int32)
            pygame.draw.rect(surf, color, [det[0], det[1], det[2] - det[0], det[3] - det[1]], 2)
            for point in det[4:].reshape(7, 2):
                pygame.draw.circle(surf, RED, [point[0], point[1]], 2)
    return pygame.surfarray.array3d(surf)


def draw_global_map(car_state, blue_array, yellow_array, midpoints, trajectory, backwards_horizon):
    res = int(config.camera_resolution / 3)
    surf = pygame.Surface((res, res))
    surf.fill(WHITE)

    car_pos = car_state[:2]
    if len(blue_array) and len(yellow_array):
        blue_array = blue_array[np.sqrt(np.square(blue_array[:, :2] - car_pos).sum(axis=1)) < config.global_map_distance]
        yellow_array = yellow_array[np.sqrt(np.square(yellow_array[:, :2] - car_pos).sum(axis=1)) < config.global_map_distance]
        if len(midpoints):
            midpoints = midpoints[np.sqrt(np.square(midpoints[:, 1:3] - car_pos).sum(axis=1)) < config.global_map_distance]

        if len(blue_array) and len(yellow_array):
            blue_array[:, :2] = rotate_around(car_pos, car_state[2], blue_array[:, :2])
            yellow_array[:, :2] = rotate_around(car_pos, car_state[2], yellow_array[:, :2])
            if len(midpoints):
                midpoints[:, 1:3] = rotate_around(car_pos, car_state[2], midpoints[:, 1:3])
            trajectory[:, :2] = rotate_around(car_pos, car_state[2], trajectory[:, :2])
            backwards_horizon[:, :2] = rotate_around(car_pos, car_state[2], backwards_horizon[:, :2])

            p1x = int(res / 2)
            scaler_x = (res / 2) / 20
            p1y = int(res / 2)
            scaler_y = (res / 2) / 40

            for cone_array, color in [(yellow_array, YELLOW), (blue_array, BLUE)]:
                for cone in cone_array:
                    pygame.draw.circle(surf, color, [p1x + int(cone[0] * scaler_x), p1y - int(cone[1] * scaler_y)], 5+5*cone[3])

            # Draw the same splines MPC will use internally:
            if len(midpoints) > 1:
                k = 1
                if len(midpoints) > 2:
                    k = config.spline_deg
                distances = config.b_spline_points
                cx_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 1], k=k)
                cy_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 2], k=k)
                # Fitting casadi bsplines on all zeros returns NaN, fix:
                c_x = interpolate.splev(distances, cx_spline) + 1e-10
                c_y = interpolate.splev(distances, cy_spline) + 1e-10
                c_dx = interpolate.splev(distances, cx_spline, der=1) + 1e-10
                c_dy = interpolate.splev(distances, cy_spline, der=1) + 1e-10

                theta_cx = casadi.interpolant("theta_cx", "linear", [distances], c_x)
                theta_cy = casadi.interpolant("theta_cy", "linear", [distances], c_y)
                theta_cdx = casadi.interpolant("theta_cdx", "linear", [distances], c_dx)
                theta_cdy = casadi.interpolant("theta_cdy", "linear", [distances], c_dy)
                cx = np.asarray(theta_cx(distances))
                cy = np.asarray(theta_cy(distances))
                cdx = np.asarray(theta_cdx(distances))
                cdy = np.asarray(theta_cdy(distances))

                for (x, y, dx, dy) in zip(cx, cy, cdx, cdy):
                    pygame.draw.circle(surf, DEBUG, [p1x + int(x * scaler_x), p1y - int(y * scaler_y)], 2)
                    angle = np.arctan2(dy, dx)
                    mp_x_b = x + 1*np.cos(angle)
                    mp_y_b = y + 1*np.sin(angle)
                    pygame.draw.line(surf, DEBUG,
                                     [p1x + int(x * scaler_x), p1y - int(y * scaler_y)],
                                     [p1x + int(mp_x_b * scaler_x), p1y - int(mp_y_b * scaler_y)],
                                     2)

            cmap = plt.colormaps["cool"]
            if len(trajectory):
                for pt in trajectory:
                    color = tuple([int(255*c) for c in cmap(1/config.car_max_speed*pt[2])[:3]])
                    pygame.draw.circle(surf, color, [p1x + int(pt[0] * scaler_x), p1y - int(pt[1] * scaler_y)], 5)
            for pt in backwards_horizon:
                color = tuple([int(255*c) for c in cmap(1/config.car_max_speed*pt[2])[:3]])
                pygame.draw.circle(surf, color, [p1x + int(pt[0] * scaler_x), p1y - int(pt[1] * scaler_y)], 5)

            x_car = p1x
            y_car = p1y
            pygame.draw.circle(surf, BLACK, [x_car, y_car], 5)
            p2x = int(x_car + np.cos(np.deg2rad(config.camera_fov/2) + np.deg2rad(90)) * scaler_x * config.view_distance)
            p2y = int(y_car - np.sin(np.deg2rad(config.camera_fov/2) + np.deg2rad(90)) * scaler_y * config.view_distance)
            p3x = int(x_car + np.cos(-np.deg2rad(config.camera_fov/2) + np.deg2rad(90)) * scaler_x * config.view_distance)
            p3y = int(y_car - np.sin(-np.deg2rad(config.camera_fov/2) + np.deg2rad(90)) * scaler_y * config.view_distance)
            pygame.draw.line(surf, BLACK, [x_car, y_car], [p2x, p2y], 5)
            pygame.draw.line(surf, BLACK, [x_car, y_car], [p3x, p3y], 5)

    return pygame.surfarray.array3d(surf)


