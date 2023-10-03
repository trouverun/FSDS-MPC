class PerceptionData:
    camera_frame = None
    blue_pixels = None
    yellow_pixels = None
    camera_ms = None
    yolo_ms = None
    kp_ms = None
    pnp_ms = None
    map_update_ms = None


class ControllerOutput:
    timestamp = None
    trajectory = None
    steer = None
    throttle = None
    d_steer = None
    solver_success = None
    stop = False


class ControllerData:
    car_state = None
    blue_cones = None
    yellow_cones = None
    midpoints = None
    trajectory = None
    solver_time_ms = None


