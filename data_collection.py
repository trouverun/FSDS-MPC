import logging
import time
import scipy
import argparse
import numpy as np
import shutil
import config
from casadi import *
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from driving_system.queue_messages import ControllerOutput
from utils.sim_helper import SimHelper
from utils.client_helper import ClientHelper
from utils.data_storage import DataStorage


N = 50


def get_solver():
    s = SX.sym('s')
    # set up states & controls
    v = SX.sym('v')
    t = SX.sym('t')
    x = vertcat(v, t)
    t_d = SX.sym('t_d')
    u = vertcat(t_d)
    ax = (
            ((config.drivetrain_params["car_Tm0"] + config.drivetrain_params["car_Tm1"] * (v/config.car_max_speed)) * t)
            - (config.drivetrain_params["car_Tr0"]*(1-tanh(config.drivetrain_params["car_Tr1"]*(v/config.car_max_speed))) + config.drivetrain_params["car_Tr2"] * (v/config.car_max_speed)**2)
    )
    f_expr = vertcat(
        ax,
        t_d
    )
    f = Function('f', [x, u], [f_expr])
    dynamics = x + 0.02 * f(x, u)
    model = AcadosModel()
    model.disc_dyn_expr = dynamics
    model.x = x
    model.u = u
    model.p = vertcat(s)
    model.name = "drivetrain"
    ocp = AcadosOcp()
    ocp.dims.N = N
    ocp.model = model
    ocp.constraints.lbu = np.array([0])
    ocp.constraints.ubu = np.array([0.25])
    ocp.constraints.idxbu = np.arange(model.u.size()[0])
    ocp.constraints.lbx = np.array([-0.5])
    ocp.constraints.ubx = np.array([0.5])
    ocp.constraints.idxbx = np.array([1])
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.model.cost_y_expr = vertcat(v - s)
    ocp.cost.yref = np.array([0])
    ocp.cost.W = scipy.linalg.block_diag(5)
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.model.cost_y_expr_e = ocp.model.cost_y_expr
    ocp.cost.yref_e = ocp.cost.yref
    ocp.cost.W_e = ocp.cost.W
    # Initial state constraint:
    ocp.constraints.x0 = np.zeros(model.x.size()[0])
    ocp.parameter_values = np.zeros(model.p.size()[0])
    ocp.solver_options.tf = 1
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hpipm_mode = 'SPEED'
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_iter_max = 50
    ocp.solver_options.nlp_solver_max_iter = 10
    ocp.solver_options.tol = 1e-3
    return AcadosOcpSolver(ocp, json_file="acados_ocp.json")

parser = argparse.ArgumentParser()
parser.add_argument('--sim_path', type=str)
parser.add_argument('--experiments', type=str, nargs="*")
args = parser.parse_args()

logger = logging.getLogger("sine_data_gen")
sim_helper = SimHelper(args.sim_path, logger)
acados_solver = get_solver()

map_name = "Null_a"
for experiment in args.experiments:
    values2 = [None]
    if experiment == 'a':
        limit = 20
        data_storage = DataStorage('acceleration_data')
        values1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    elif experiment == 's_r':
        limit = 15
        data_storage = DataStorage('all_in_one_v2')
        values1 = [7, 9, 11]
    elif experiment == 's_rl':
        limit = 50
        data_storage = DataStorage('all_in_one_v2')
        values1 = [11, 13, 15]
    elif experiment == 's_s':
        limit = 20
        data_storage = DataStorage('all_in_one_v2')
        values1 = [8, 10, 12, 14]
        values2 = [0.25, 0.5, 0.75]
    else:
        continue

    for value1 in values1:
        for value2 in values2:
            sim_helper.start_sim(map_name)
            client_helper = ClientHelper(logger, controller=True)
            steer, d_steer, throttle = 0, 0, 0
            times = []
            initialized = False
            x0 = np.zeros([N+1, 2])
            u0 = np.zeros([N, 1])
            started = False
            offset = 0
            while True:
                t1 = time.time_ns()
                car_state = client_helper.read_car_state()
                ref_state = client_helper.read_referee_state()

                if (ref_state.time_passed_s-offset) > limit:
                    break
                if experiment == 'a':
                    steer = 0
                    d_steer = 0
                    period = value1*2*np.pi / config.u_steer_max
                    throttle = value1/2 + value1/2 * np.sin(ref_state.time_passed_s / period * 2 * np.pi + np.arcsin(-1))
                elif experiment in ['s_r', 's_rl', 's_s']:
                    initial_state = np.array([car_state.car_linear_vel[0], throttle])
                    acados_solver.set(0, "lbx", initial_state)
                    acados_solver.set(0, "ubx", initial_state)

                    for i in range(N + 1):
                        acados_solver.set(i, "p", np.array([value1]))
                        if initialized:
                            if i == 0:
                                acados_solver.set(i, "x", initial_state)
                            else:
                                acados_solver.set(i, "x", x0[i])
                            if i < N:
                                acados_solver.set(i, "u", u0[i])
                        else:
                            acados_solver.set(i, "x", initial_state)

                    _ = acados_solver.solve()

                    for i in range(N + 1):
                        x = acados_solver.get(i, "x")
                        x0[i] = x
                        if i < N:
                            u = acados_solver.get(i, "u")
                            u0[i] = u
                            if i == 0:
                                d_steer = u[0]
                        if i == 1:
                            throttle = x[1]
                            throttle = np.clip(throttle, -1, 1)
                    x0[:-1] = x0[1:]
                    u0[:-1] = u0[1:]

                    if not started and abs(car_state.car_linear_vel[0] - value1) < 1e-1:
                        started = True
                        offset = ref_state.time_passed_s

                    if started:
                        if experiment in ['s_r', 's_rl']:
                            steer = (ref_state.time_passed_s-offset)*(0.5/np.rad2deg(config.car_max_steer))
                            d_steer = np.deg2rad(0.5)
                        elif experiment == 's_s':
                            period = value2 * 2 * np.pi / config.u_steer_max
                            steer = value2 * np.sin((ref_state.time_passed_s - offset) / period * 2 * np.pi)
                            d_steer = value2 * 2 * np.pi / period * np.cos((ref_state.time_passed_s - offset) / period * 2 * np.pi)

                steer = np.clip(steer, -1, 1).item()
                client_helper.set_controls(steer, throttle, d_steer)

                state = np.array([
                    car_state.timestamp,  # 0
                    car_state.car_pos[0],  # 1
                    car_state.car_pos[1],  # 2
                    car_state.car_hdg,  # 3
                    car_state.car_linear_vel[0],  # 4
                    car_state.car_linear_vel[1],  # 5
                    car_state.car_angular_vel,  # 6
                    car_state.car_linear_acc[0],  # 7
                    car_state.car_linear_acc[1],  # 8
                    0,  # 9 (dw needs to be calculated with finite diff)
                    steer,  # 10
                    throttle,  # 11
                    d_steer,  # 12
                ])

                data_storage.record_data(state)

                t2 = time.time_ns()
                time.sleep(max(0.02 - (t2-t1)/1e9, 0))

            sim_helper.stop_sim()
            data_storage.save_dataset()

