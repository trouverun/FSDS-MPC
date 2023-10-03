from casadi import *
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
import config
import scipy

fake_inf = 1E7


def smooth_min_max(val1, val2, alpha):
    return (val1*exp(alpha*val1) + val2*exp(alpha*val2)) / (exp(alpha*val1) + exp(alpha*val2))


def create_dynamic_bicycle_model(ocp, casadi_torch_model=None):
    ocp.model = AcadosModel()
    ocp.model.name = "dynamic_bicycle"

    # PARAMETERS ------------------------------------------------------------------------------------------------------

    # State
    x = MX.sym("x")
    y = MX.sym("y")
    hdg = MX.sym("hdg")
    w = MX.sym("w")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    steer = MX.sym("steer")
    throttle = MX.sym("throttle")
    theta = MX.sym("theta")
    ocp.model.x = vertcat(x, y, hdg, vx, vy, w, steer, throttle, theta)

    # Controls
    u_steer = MX.sym("usteer")
    u_throttle = MX.sym("uthrottle")
    u_theta = MX.sym("utheta")
    ocp.model.u = vertcat(u_steer, u_throttle, u_theta)

    # Configurable params
    dt = MX.sym("dt", 1)

    # Model params
    car_Tm0 = MX.sym("car_Tm0")
    car_Tm1 = MX.sym("car_Tm1")
    car_Tm2 = MX.sym("car_Tm2")
    car_Tr0 = MX.sym("car_Tr0")
    car_Tr1 = MX.sym("car_Tr1")
    car_Tr2 = MX.sym("car_Tr2")
    drivetrain_params = vertcat(car_Tm0, car_Tm1, car_Tm2, car_Tr0, car_Tr1, car_Tr2)
    car_inertia = MX.sym("car_inertia")
    wheel_Bf = MX.sym("wheel_Bf")
    wheel_Cf = MX.sym("wheel_Cf")
    wheel_Df = MX.sym("wheel_Df")
    wheel_Br = MX.sym("wheel_Br")
    wheel_Cr = MX.sym("wheel_Cr")
    wheel_Dr = MX.sym("wheel_Dr")
    bicycle_params = vertcat(car_inertia, wheel_Bf, wheel_Cf, wheel_Df, wheel_Br, wheel_Cr, wheel_Dr)

    # Parametric splines for the track center line:
    theta0 = config.b_spline_points
    # center x position
    theta_cx = interpolant("theta_cx", "linear", [theta0])
    cx0 = MX.sym("cx0", config.n_bspline_points)
    cx_interp_exp = theta_cx(theta, cx0)
    cx_fun = Function('cx_fun', [theta, cx0], [cx_interp_exp])
    # center y position
    theta_cy = interpolant("theta_cy", "linear", [theta0])
    cy0 = MX.sym("cy0", config.n_bspline_points)
    cy_interp_exp = theta_cy(theta, cy0)
    cy_fun = Function('cy_fun', [theta, cy0], [cy_interp_exp])
    # center dx:
    theta_cdx = interpolant("theta_cdx", "linear", [theta0])
    cdx0 = MX.sym("cdx0", config.n_bspline_points)
    cdx_interp_exp = theta_cdx(theta, cdx0)
    cdx_fun = Function('cxd_fun', [theta, cdx0], [cdx_interp_exp])
    # center dy:
    theta_cdy = interpolant("theta_cdy", "linear", [theta0])
    cdy0 = MX.sym("cdy0", config.n_bspline_points)
    cdy_interp_exp = theta_cdy(theta, cdy0)
    cdy_fun = Function('cdy_fun', [theta, cdy0], [cdy_interp_exp])

    # Model parameters
    ocp.model.z = vertcat([])
    ocp.model.p = vertcat(cx0, cy0, cdx0, cdy0, dt, drivetrain_params, bicycle_params)

    # DYNAMICS --------------------------------------------------------------------------------------------------------

    input_state = vertcat(hdg, vx, vy, w, steer, throttle)

    af = -atan2(w*config.car_lf + vy, vx+0.01) + steer*config.car_max_steer
    Ffy = wheel_Df * sin(wheel_Cf * arctan(wheel_Bf * af))

    ar = atan2(w*config.car_lr - vy, vx+0.01)
    Fry = wheel_Dr * sin(wheel_Cr * arctan(wheel_Br * ar))

    Frx = config.car_mass * (
            ((car_Tm0 + car_Tm1 * (vx/config.car_max_speed) + car_Tm2 * (vx/config.car_max_speed) * throttle) * throttle)
            - (car_Tr0*(1-tanh(car_Tr1*(vx/config.car_max_speed))) + car_Tr2 * (vx/config.car_max_speed)**2)
    )
    f_d_expr = vertcat(
        vx * cos(hdg) - vy*sin(hdg),
        vx * sin(hdg) + vy*cos(hdg),
        w,
        1 / config.car_mass * (Frx - Ffy * sin(steer * config.car_max_steer) + config.car_mass*vy*w),
        1 / config.car_mass * (Fry + Ffy * cos(steer * config.car_max_steer) - config.car_mass*vx*w),
        1 / car_inertia * (Ffy*config.car_lf*cos(steer*config.car_max_steer) - Fry*config.car_lr),
        0,
        0
    )
    f_k_expr = vertcat(
        vx * cos(hdg) - vy * sin(hdg),
        vx * sin(hdg) + vy * cos(hdg),
        w,
        Frx/config.car_mass,
        (u_steer*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (config.car_lr/(config.car_lr + config.car_lf)),
        (u_steer*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (1 / (config.car_lr + config.car_lf)),
        0,
        0
    )
    vb_min = config.blend_min_speed
    vb_max = config.blend_max_speed

    # lam = fmin(fmax((vx - vb_min) / (vb_max - vb_min), 0), 1)
    lam = smooth_min_max(smooth_min_max((vx - vb_min) / (vb_max - vb_min), 0, 10), 1, -10)

    f_expr = lam*f_d_expr + (1-lam)*f_k_expr
    f = Function('f', [input_state, u_steer, drivetrain_params, bicycle_params], [f_expr])
    # Discretize with RK4:
    k1 = f(input_state, u_steer, drivetrain_params, bicycle_params)
    # crop the x and y since they are part of the output, but not part of the input
    k2 = f(input_state + dt / 2 * k1[2:], u_steer, drivetrain_params, bicycle_params)
    k3 = f(input_state + dt / 2 * k2[2:], u_steer, drivetrain_params, bicycle_params)
    k4 = f(input_state + dt * k3[2:], u_steer, drivetrain_params, bicycle_params)
    dynamics = dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    dynamics = dynamics[:-2]  # remove steer and throttle since they are dealt by simple euler dynamics below
    delay_compensation_f = Function("delay_comp", [input_state, u_steer, drivetrain_params, bicycle_params, dt], [dynamics])

    # Shared dynamics, control states are discretized with simple euler:
    ocp.model.disc_dyn_expr = ocp.model.x + vertcat(
        dynamics,
        dt * u_steer,
        dt * u_throttle,
        dt * u_theta,
    )

    # CONSTRAINTS -----------------------------------------------------------------------------------------------------

    # State bounds
    x_min = -fake_inf
    x_max = fake_inf
    y_min = -fake_inf
    y_max = fake_inf
    hdg_min = -fake_inf
    hdg_max = fake_inf
    w_min = -fake_inf
    w_max = fake_inf
    vx_min = 0
    vx_max = config.car_max_speed
    vy_min = -fake_inf
    vy_max = fake_inf
    steer_min = -config.steer_value_max
    steer_max = config.steer_value_max
    throttle_min = -config.throttle_value_max
    throttle_max = config.throttle_value_max
    theta_min = -fake_inf
    theta_max = fake_inf
    ocp.constraints.lbx = np.array([x_min, y_min, hdg_min, vx_min, vy_min, w_min, steer_min, throttle_min, theta_min])
    ocp.constraints.lbx_e = ocp.constraints.lbx
    ocp.constraints.ubx = np.array([x_max, y_max, hdg_max, vx_max, vy_max, w_max, steer_max, throttle_max, theta_max])
    ocp.constraints.ubx_e = ocp.constraints.ubx
    ocp.constraints.idxbx = np.arange(ocp.model.x.size()[0])
    ocp.constraints.idxbx_e = ocp.constraints.idxbx
    # Soft constraints:
    ocp.constraints.lsbx = np.array([0])
    ocp.constraints.lsbx_e = ocp.constraints.lsbx
    ocp.constraints.usbx = np.array([0])
    ocp.constraints.usbx_e = ocp.constraints.usbx
    ocp.constraints.idxsbx = np.array([3])
    ocp.constraints.idxsbx_e = ocp.constraints.idxsbx
    state_slack_weights = np.array([config.soft_vx_slack_weight])

    # Control bounds:
    u_steer_min = -config.u_steer_max
    u_steer_max = config.u_steer_max
    u_throttle_min = -config.u_throttle_max
    u_throttle_max = config.u_throttle_max
    u_theta_min = -config.u_theta_max
    u_theta_max = config.u_theta_max
    ocp.constraints.lbu = np.array([u_steer_min, u_throttle_min, u_theta_min])
    ocp.constraints.ubu = np.array([u_steer_max, u_throttle_max, u_theta_max])
    ocp.constraints.idxbu = np.arange(ocp.model.u.size()[0])
    # Soft constraints:
    ocp.constraints.lsbu = np.array([])
    ocp.constraints.usbu = np.array([])
    ocp.constraints.idxsbu = np.array([])

    # Nonlinear constraints
    ocp.constraints.lh = np.array([0, 0])
    ocp.constraints.lh_e = np.array([0, 0])
    ocp.constraints.uh = np.array([config.track_radius**2, 1])
    ocp.constraints.uh_e = np.array([config.track_radius**2, 0])
    center_circle_deviation = (x - cx_fun(theta, cx0))**2 + (y - cy_fun(theta, cy0))**2
    ax = 1 / config.car_mass * (Frx - Ffy * sin(steer * config.car_max_steer))
    ay = 1 / config.car_mass * (Fry + Ffy * cos(steer * config.car_max_steer))
    ocp.model.con_h_expr = vertcat(center_circle_deviation, (ax/config.ax_max)**2+(ay/config.ax_max)**2)
    ocp.model.con_h_expr_e = vertcat(center_circle_deviation, ay)
    # Soft constraints
    ocp.constraints.lsh = np.array([0, 0])
    ocp.constraints.ush = np.array([0, 0])
    ocp.constraints.idxsh = np.array([0, 1])
    ocp.constraints.ush_e = np.array([0, 0])
    ocp.constraints.lsh_e = np.array([0, 0])
    ocp.constraints.idxsh_e = np.array([0, 1])
    nonlinear_slack_weights = np.array([config.soft_nl_track_circle_weight, config.soft_nl_accel_weight])
    nonlinear_slack_weights_e = np.array([config.soft_nl_track_circle_weight, config.soft_nl_accel_weight])

    # COSTS -----------------------------------------------------------1------------------------------------------------

    # Constraint hessian and diagonal
    ocp.cost.zl = np.zeros(len(state_slack_weights) + len(nonlinear_slack_weights))
    ocp.cost.zl_e = np.zeros(len(state_slack_weights) + len(nonlinear_slack_weights_e))
    ocp.cost.zu = np.zeros(len(state_slack_weights) + len(nonlinear_slack_weights))
    ocp.cost.zu_e = np.zeros(len(state_slack_weights) + len(nonlinear_slack_weights_e))
    ocp.cost.Zl = np.r_[fake_inf*state_slack_weights, nonlinear_slack_weights]
    ocp.cost.Zl_e = np.r_[fake_inf*state_slack_weights, nonlinear_slack_weights_e]
    ocp.cost.Zu = np.r_[state_slack_weights, nonlinear_slack_weights]
    ocp.cost.Zu_e = np.r_[state_slack_weights, nonlinear_slack_weights_e]

    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    phi = atan2(cdy_fun(theta, cdy0), cdx_fun(theta, cdx0))
    e_contour = sin(phi) * (x - cx_fun(theta, cx0)) - cos(phi) * (y - cy_fun(theta, cy0))
    e_lag = -cos(phi) * (x - cx_fun(theta, cx0)) - sin(phi) * (y - cy_fun(theta, cy0))
    ocp.model.cost_y_expr = vertcat(e_contour, e_lag, u_theta, u_steer, u_throttle)
    ocp.model.cost_y_expr_e = vertcat(e_contour, e_lag)
    ocp.cost.yref = np.array([0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0])
    ocp.cost.W = scipy.linalg.block_diag(
        config.contour_weight, config.lag_weight, -config.theta_weight,
        config.u_steer_weight, config.u_throttle_weight
    )
    ocp.cost.W_e = config.mpc_dt*scipy.linalg.block_diag(config.contour_weight, config.lag_weight)

    return delay_compensation_f