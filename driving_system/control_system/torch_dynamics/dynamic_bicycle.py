import config
import torch
import numpy as np


class DriveTrain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.car_Tm0 = torch.nn.Parameter(torch.tensor([config.drivetrain_params["car_Tm0"]]))
        self.car_Tm1 = torch.nn.Parameter(torch.tensor([config.drivetrain_params["car_Tm1"]]))
        self.car_Tm2 = torch.nn.Parameter(torch.tensor([config.drivetrain_params["car_Tm2"]]))
        self.car_Tr0 = torch.nn.Parameter(torch.tensor([config.drivetrain_params["car_Tr0"]]))
        self.car_Tr1 = torch.nn.Parameter(torch.tensor([config.drivetrain_params["car_Tr1"]]))
        self.car_Tr2 = torch.nn.Parameter(torch.tensor([config.drivetrain_params["car_Tr2"]]))

    def forward(self, tensor):
        vx = tensor[:, 0]
        throttle = tensor[:, 1]
        ax = (
                ((self.car_Tm0 + self.car_Tm1 * (vx/config.car_max_speed) + self.car_Tm2 * (vx/config.car_max_speed) * throttle) * throttle)
                - (self.car_Tr0*(1-torch.tanh(self.car_Tr1*(vx/config.car_max_speed))) + self.car_Tr2 * (vx/config.car_max_speed)**2)
        )
        return torch.vstack([ax]).T


class DynamicBicycle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.car_inertia = torch.nn.Parameter(torch.tensor([config.bicycle_params["car_inertia"]]))
        self.wheel_Bf = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Bf"]]))
        self.wheel_Cf = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Cf"]]))
        self.wheel_Df = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Df"]]))
        self.wheel_Br = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Br"]]))
        self.wheel_Cr = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Cr"]]))
        self.wheel_Dr = torch.nn.Parameter(torch.tensor([config.bicycle_params["wheel_Dr"]]))
        self.drivetrain = DriveTrain()

    def extract_params(self):
        drivetrain_params = np.array([
            self.drivetrain.car_Tm0.cpu().item(),
            self.drivetrain.car_Tm1.cpu().item(),
            self.drivetrain.car_Tm2.cpu().item(),
            self.drivetrain.car_Tr0.cpu().item(),
            self.drivetrain.car_Tr1.cpu().item(),
            self.drivetrain.car_Tr2.cpu().item(),
        ])
        bicycle_params = np.array([
            self.car_inertia.cpu().item(),
            self.wheel_Bf.cpu().item(),
            self.wheel_Cf.cpu().item(),
            self.wheel_Df.cpu().item(),
            self.wheel_Br.cpu().item(),
            self.wheel_Cr.cpu().item(),
            self.wheel_Dr.cpu().item()
        ])
        return drivetrain_params, bicycle_params

    def forward(self, tensor):
        hdg = tensor[:, 0]
        vx = tensor[:, 1]
        vy = tensor[:, 2]
        w = tensor[:, 3]
        steer = tensor[:, 4]
        throttle = tensor[:, 5]
        steer_dot = tensor[:, 6]

        af = -torch.arctan2(w*config.car_lf + vy, vx+0.01) + steer*config.car_max_steer
        Ffy = self.wheel_Df * torch.sin(self.wheel_Cf * torch.arctan(self.wheel_Bf * af))

        ar = torch.arctan2(w*config.car_lr - vy, vx+0.01)
        Fry = self.wheel_Dr * torch.sin(self.wheel_Cr * torch.arctan(self.wheel_Br * ar))
        #Fry = torch.zeros_like(Fry).to(Fry.device)

        drivetrain_tensor = torch.vstack([vx, throttle]).T
        Frx = config.car_mass * self.drivetrain(drivetrain_tensor)[:, 0]
        out_d = torch.vstack([
             vx * torch.cos(hdg) - vy * torch.sin(hdg),
             vx * torch.sin(hdg) + vy * torch.cos(hdg),
             w,
             1 / config.car_mass * (Frx - Ffy * torch.sin(steer * config.car_max_steer) + config.car_mass*vy*w),
             1 / config.car_mass * (Fry + Ffy * torch.cos(steer * config.car_max_steer) - config.car_mass*vx*w),
             1 / self.car_inertia * (Ffy * config.car_lf * torch.cos(steer * config.car_max_steer) - Fry * config.car_lr)
        ]).T

        out_k = torch.vstack([
            vx * torch.cos(hdg) - vy * torch.sin(hdg),
            vx * torch.sin(hdg) + vy * torch.cos(hdg),
            w,
            Frx/config.car_mass,
            (steer_dot*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (config.car_lr / (config.car_lr + config.car_lf)),
            (steer_dot*config.car_max_steer*vx + steer*config.car_max_steer*(Frx/config.car_mass)) * (1 / (config.car_lr + config.car_lf)),
        ]).T

        vb_min = config.blend_min_speed
        vb_max = config.blend_max_speed
        lam = torch.minimum(torch.maximum((vx-vb_min) / (vb_max-vb_min), torch.zeros_like(vx)), torch.ones_like(vx))

        return (lam*out_d.T).T + ((1-lam)*out_k.T).T









