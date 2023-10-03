from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from driving_system.control_system.dynamics_models.dynamic_bicycle import create_dynamic_bicycle_model
import numpy as np
import config
from scipy import interpolate


class Solver:
    def __init__(self, dynamics_type, casadi_torch_model=None):
        self.output_root = 'debug'
        self.N = config.mpc_horizon
        self.dynamics_type = dynamics_type
        self.n_controls = 0
        self.n_states = 0
        self.n_parameters = 0

        self.ocp = None
        self.acados_solver = None
        self.delay_comp_fun = None
        self._create_solver(casadi_torch_model)

        self.initialized = False
        self.x0 = np.zeros([self.N+1, self.n_states])
        self.u0 = np.zeros([self.N, self.n_controls])
        self.last_drivetrain_params = None
        self.last_bicycle_params = None

    def _create_solver(self, casadi_torch_model):
        self.ocp = AcadosOcp()
        self.ocp.dims.N = self.N

        if self.dynamics_type not in ['dynamic_bicycle']:
            raise ValueError("unknown dynamics model type %s" % self.dynamics_type)

        if self.dynamics_type == 'dynamic_bicycle':
            self.delay_comp_fun = create_dynamic_bicycle_model(self.ocp, casadi_torch_model)

        self.n_controls = self.ocp.model.u.size()[0]
        self.n_states = self.ocp.model.x.size()[0]
        self.n_parameters = self.ocp.model.p.size()[0]

        self.ocp.constraints.x0 = np.zeros(self.n_states)
        self.ocp.parameter_values = np.zeros(self.n_parameters)

        self.ocp.solver_options.tf = config.mpc_dt * config.mpc_horizon
        self.ocp.solver_options.integrator_type = "DISCRETE"
        self.ocp.solver_options.nlp_solver_type = "SQP_RTI"
        self.ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # self.ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        self.ocp.qp_solver_warm_start = 1
        # self.ocp.solver_options.hpipm_mode = 'ROBUST'
        self.ocp.solver_options.hpipm_mode = 'SPEED'
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        self.ocp.solver_options.regularize_method = "PROJECT_REDUC_HESS"
        # self.ocp.solver_options.regularize_method = "CONVEXIFY"
        # self.ocp.solver_options.qp_solver_cond_ric_alg = 1
        # self.ocp.solver_options.line_search_use_sufficient_descent = 1
        self.ocp.solver_options.levenberg_marquardt = 0.01
        #self.ocp.solver_options.qp_solver_iter_max = config.qp_solver_max_iter
        #self.ocp.solver_options.nlp_solver_max_iter = config.nlp_solver_max_iter
        #self.ocp.solver_options.tol = config.solver_tolerance
        self.ocp.solver_options.print_level = 0

        self.acados_solver = AcadosOcpSolver(self.ocp, json_file="acados_ocp.json")

    def delay_compensation(self, state, d_steer, drivetrain_params, bicycle_params, dt):
        update = self.delay_comp_fun(state[2:], d_steer, drivetrain_params, bicycle_params, dt)
        update = np.asarray(update).flatten()
        updated_state = state
        updated_state[:-2] += update

        self.last_drivetrain_params = drivetrain_params
        self.last_bicycle_params = bicycle_params

        return updated_state

    def initialize(self, initial_state, midpoints, max_speed, dt, params):
        initial_state = np.r_[initial_state.flatten(), 0]
        self.acados_solver.set(0, "lbx", initial_state)
        self.acados_solver.set(0, "ubx", initial_state)

        k = 1
        if len(midpoints) > 2:
            k = config.spline_deg
        distances = config.b_spline_points
        cx_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 1], k=k)
        cy_spline = interpolate.splrep(midpoints[:, 0], midpoints[:, 2], k=k)
        c_x = interpolate.splev(distances, cx_spline)
        c_y = interpolate.splev(distances, cy_spline)
        c_dx = interpolate.splev(distances, cx_spline, der=1)
        c_dy = interpolate.splev(distances, cy_spline, der=1)

        # Fill in the spline parameters, model approximation parameters and initial guesses
        for i in range(config.mpc_horizon + 1):
            p = np.r_[c_x, c_y, c_dx, c_dy, dt[i], params]
            self.acados_solver.set(i, "p", p)

            if i > 0:
                ubx = self.ocp.constraints.ubx
                ubx[3] = max_speed
                self.acados_solver.set(i, "ubx", ubx)

            if self.initialized:
                if i == 0:
                    self.acados_solver.set(i, "x", initial_state)
                else:
                    self.acados_solver.set(i, "x", self.x0[i])
                if i < config.mpc_horizon:
                    self.acados_solver.set(i, "u", self.u0[i])
            else:
                self.acados_solver.set(i, "x", initial_state)

    def _shift_horizon(self):
        self.x0[:-1] = self.x0[1:]
        self.u0[:-1] = self.u0[1:]
        if self.last_drivetrain_params is not None and self.last_bicycle_params is not None:
            self.x0[-1, :-1] = self.delay_compensation(self.x0[-1, :-1], self.u0[-1, 0], self.last_drivetrain_params, self.last_bicycle_params, 1/10)

    def solve(self):
        for i in range(config.nlp_solver_max_iter):
            status = self.acados_solver.solve()

        steer, throttle, d_steer = 0, -1, 0
        if status in [0, 2]:   # Success or timeout
            self.initialized = True
            for i in range(config.mpc_horizon+1):
                x = self.acados_solver.get(i, "x")
                self.x0[i] = x
                if i < config.mpc_horizon:
                    u = self.acados_solver.get(i, "u")
                    self.u0[i] = u
                    if i == 0:
                        d_steer = u[0]
                if i == 1:
                    steer = x[6]
                    throttle = x[7]
                    # We might terminate early with some constraint violations, clip to make sure:
                    steer = np.clip(steer, -1, 1)
                    throttle = np.clip(throttle, -1, 1)
            horizon = self.x0.copy()
            self._shift_horizon()
            return steer, throttle, horizon, d_steer
        else:
            self._shift_horizon()
            raise RuntimeError("Solver failed with status %d" % status)