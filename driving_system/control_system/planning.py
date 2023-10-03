import numpy as np
import config


def find_centerline(car_pos, car_hdg, blue_array, yellow_array, no_path_limit=False):
    matches = []
    path = []
    # Iterate through the shorter cone array, and match each cone with the nearest cone of opposing color.
    # From each match calculate a track midpoint.
    if len(blue_array) <= len(yellow_array):
        # Start matching from the nearest blue cone
        dist = np.sqrt(np.square(blue_array[:, :2] - car_pos[:2]).sum(axis=1))
        blue_array = blue_array[np.argsort(dist)]
        for point in blue_array:
            available_yellow = [*range(0, len(yellow_array))]
            dist = np.sqrt(np.square((yellow_array[:, :2] - point[:2])).sum(axis=1))
            matched_yellow_i = np.argmin(dist)
            # If the distance between cones is large, then they are not even close to being correct correspondences, and we ignore them
            if config.min_cone_separation < dist[matched_yellow_i] < config.max_cone_separation:
                matches.append(
                    np.concatenate([
                        (point + yellow_array[matched_yellow_i])[:2] / 2,
                        point[:2], yellow_array[matched_yellow_i, :2]
                    ], axis=0))
                # remove matched cone, to avoid matching same cone multiple times
                available_yellow.pop(matched_yellow_i)
                yellow_array = yellow_array[available_yellow]
    else:
        # Start matching from the nearest yellow cone
        dist = np.sqrt(np.square(yellow_array[:, :2] - car_pos[:2]).sum(axis=1))
        yellow_array = yellow_array[np.argsort(dist)]
        for point in yellow_array:
            available_blue = [*range(0, len(blue_array))]
            dist = np.sqrt(np.square((blue_array[:, :2] - point[:2])).sum(axis=1))
            matched_blue_i = np.argmin(dist)
            # If the distance between cones is large, then they are not even close to being correct correspondences, and we ignore them
            if config.min_cone_separation < dist[matched_blue_i] < config.max_cone_separation:
                matches.append(
                    np.concatenate([
                        (point + blue_array[matched_blue_i])[:2] / 2,
                        blue_array[matched_blue_i, :2], point[:2]
                    ], axis=0))
                # remove matched cone, to avoid matching same cone multiple times
                available_blue.pop(matched_blue_i)
                blue_array = blue_array[available_blue]

    # Iterate through all midpoints, and find the track path by looking for the nearest midpoint, starting from the car position.
    # Path is constrained so that blue cones are to the left, and yellow cones to the right
    if len(matches):
        matches = np.asarray(matches)
        current = car_pos[:2].copy()
        path_len = 0
        backwards_pass = True
        num_backwards = 0
        while len(matches) and (path_len < config.max_path_length or no_path_limit):
            # Calculate the angle at which we travel through the midpoint, and based on that create a constraint/weight vector:
            displacement = matches[:, :2] - current
            angles = np.arctan2(displacement[:, 1], displacement[:, 0])
            weights = np.array([-np.sin(angles), np.cos(angles), np.sin(angles), -np.cos(angles)]).T

            if backwards_pass:
                # We want to have one midpoint behind the car, it makes the midpoint splines better behaved
                values = (matches[:, 2:] * -weights).sum(axis=1)
            else:
                # Find the valid midpoints in front of the car
                values = (matches[:, 2:] * weights).sum(axis=1)

            dist = np.sqrt(np.square(matches[:, :2] - current).sum(axis=1))
            # Filter out points far away as they would likely form an incorrect path
            valid_indices = np.where(np.all([values > 0, dist < config.max_midpoint_distance], axis=0))[0]

            # If no valid midpoints found, time to exit
            if not len(valid_indices):
                if backwards_pass:
                    # If we were searching for first midpoint behind the car, just project a point directly behind the car instead:
                    if num_backwards == 0:
                        angle = car_hdg
                        path.append(np.r_[-2.5, np.array([car_pos[0] - 2.5*np.cos(angle), car_pos[1] - 2.5*np.sin(angle)])])
                    path.reverse()
                    path_len = 0
                    current = car_pos[:2].copy()
                    backwards_pass = False
                    continue
                break

            order = np.argsort(dist)
            valid_order = np.intersect1d(order, valid_indices)
            best = valid_order[0]

            # Find some points behind the car, seems to "stabilize" the splines when in sharp corners
            if backwards_pass:
                path_len -= dist[best]
                current = matches[best, :2]
                path.append(np.r_[path_len, current])
                available_indices = [*range(0, len(matches))]
                available_indices.pop(best)
                matches = matches[available_indices]
                num_backwards += 1
                if num_backwards == 3:
                    path.reverse()
                    path_len = 0
                    current = car_pos[:2].copy()
                    backwards_pass = False
                continue

            path_len += dist[best]
            current = matches[best, :2]
            path.append(np.r_[path_len, current])
            available_indices = [*range(0, len(matches))]
            available_indices.pop(best)
            matches = matches[available_indices]

    return np.asarray(path)