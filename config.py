import torch
import json
import numpy as np
from scipy.spatial.transform import Rotation
from utils.maths import trans_inv, rp_to_trans

state_estimator_hz = 50
state_estimator_dt = 1/state_estimator_hz
car_shared_state_size = 12

# Visualization
global_map_distance = 40

# Simulator
max_fsds_client_attempts = 10
max_failed_mpc_solves = 10
max_stuck_steps = 100
lap_speed_increase = 6
max_collisions = 5
view_distance = 15

# Camera params
camera_R = Rotation.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix()
camera_p = np.array([0, -0.8, -0.3])
camera_T = trans_inv(rp_to_trans(camera_R, camera_p))
camera_name = "high_res_cam"
camera_resolution = 1280
camera_fov = 75
f = camera_resolution / (2 * np.tan(camera_fov * np.pi / 360))
c = camera_resolution / 2
K = np.array([
    [f, 0, c],
    [0, f, c],
    [0, 0, 1]
])

# Cone detection regression
kp_resize_w, kp_resize_h = 80, 80
kp_min_w, kp_min_h = 25, 25
cone_3d_points = np.array([
    [0, 0, 0.305],
    [-0.042, 0, 0.18],
    [-0.058, 0, 0.1275],
    [-0.075, 0, 0.03],
    [0.042, 0, 0.18],
    [0.058, 0, 0.1275],
    [0.075, 0, 0.03]
])
min_valid_cone_distance = 2.5
max_valid_cone_distance = 15
min_bbox_conf = 0.75
BLUE_CONE = 7
YELLOW_CONE = 2

# Mapping
n_map_cones = 1000   # max allocated cones in shared memory (per color)
cone_ekf_alpha = 0.025
cone_position_variance = 1
variance_increase_distance = 10
additional_cone_pos_variance = 0.1
mapping_vision_adjuster = 0.9
delete_threshold = 0.75
min_cone_separation = 1.5
max_cone_separation = 4.5
max_midpoint_distance = 10
max_path_length = 40

# Learning
savgol_k = 7
savgol_d_k = 6
p = 1.25
patience = 100

gp_n_data = 7500
wc = 5
ws = 1
max_dev_pct = 0.25
ww = np.ones(5)
ww[0] = 18
ww[2] = 1/4*torch.pi
ww[3] = 0.5
ww[4] = 0.5
ww = 1/(max_dev_pct*ww)

# Identified params
car_max_speed = 17.5
car_lr = 0.78
car_lf = 0.41
car_mass = 190

try:
    with open('dynamics_params/drivetrain_params.json', 'r') as infile:
        drivetrain_params = json.load(infile)
except:
    raise Exception("No drivetrain params, run identify_dynamics.py first with acceleration dataset")

try:
    with open('dynamics_params/bicycle_params.json', 'r') as infile:
        bicycle_params = json.load(infile)
except:
    raise Exception("No bicycle params, run identify_dynamics.py first with steering dataset")

blend_min_speed = 3
blend_max_speed = 5

# Car controls:
car_max_steer = np.deg2rad(25)
steer_value_max = 0.9
throttle_value_max = 0.5
u_steer_max = 1.5
u_throttle_max = 2
u_theta_max = car_max_speed
ax_max = 7.5
ay_max = 12.5

# MPC
spline_deg = 2
track_radius = 0.75
n_bspline_points = 100
bspline_point_distance = 1
bspline_max_distance = n_bspline_points*bspline_point_distance
b_spline_points = np.arange(0, bspline_max_distance, bspline_point_distance)
assert len(b_spline_points) == n_bspline_points
mpc_horizon = 40
mpc_hz = 25
mpc_dt = 1/mpc_hz
car_initial_max_speed = 6
# Weights for nonlinear lsq cost:
lag_weight = 100
contour_weight = 0.01
theta_weight = 0.1
u_steer_weight = 1
u_throttle_weight = 1
# Soft constraint violation weights:
soft_vx_slack_weight = 10
soft_nl_track_circle_weight = 100
soft_nl_accel_weight = 100
# solver params
qp_solver_max_iter = 250
nlp_solver_max_iter = 15
solver_tolerance = 1e-3