import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import config
from driving_system.control_system.torch_dynamics.dynamic_bicycle import DriveTrain, DynamicBicycle
from utils.data_storage import DataStorage
from utils.dynamics_trainer import train_fn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--type', type=str)
args = parser.parse_args()

data_storage = DataStorage(args.dataset)
original_inputs, original_outputs, filtered_inputs, filtered_outputs = data_storage.get_training_data()

if args.type == "acceleration_data":
    original_inputs = original_inputs[:, [1, 5]]
    filtered_inputs = filtered_inputs[:, [1, 5]]
    original_outputs = original_outputs[:, [3]]
    filtered_outputs = filtered_outputs[:, [3]]
    torch_model = DriveTrain().cuda()
    optimizer = torch.optim.AdamW([
        {'params': [torch_model.car_Tm0, torch_model.car_Tm1, torch_model.car_Tr1], 'lr': 1e-1, 'weight_decay': 1e-8, 'betas': (0.90, 0.999)},
        {'params': [torch_model.car_Tr0, torch_model.car_Tr2, torch_model.car_Tm2], 'lr': 1e-2, 'weight_decay': 1e-8, 'betas': (0.90, 0.999)},
    ])
    w = torch.tensor([1])
elif args.type in ["steering_data"]:
    vx = filtered_inputs[:, 1]
    vy = filtered_inputs[:, 2]
    w = filtered_inputs[:, 3]
    steer = filtered_inputs[:, 4]

    additionals = np.c_[
        np.arctan2(config.car_lr * np.tan(steer * config.car_max_steer), config.car_lr + config.car_lf),
        -np.arctan2(w * config.car_lf + vy, vx + 0.01) + steer * config.car_max_steer,
        np.arctan2(w * config.car_lr - vy, vx + 0.01)
    ]

    torch_model = DynamicBicycle().cuda()

    optimizer = torch.optim.AdamW([
        {'params': [torch_model.wheel_Dr], 'lr': 1e1, 'weight_decay': 1e-5, 'betas': (0.90, 0.999)},
        {'params': [torch_model.wheel_Df], 'lr': 1e1, 'weight_decay': 5e-4, 'betas': (0.90, 0.999)},
        {'params': [torch_model.car_inertia], 'lr': 1e1, 'weight_decay': 1e-3, 'betas': (0.90, 0.999)},
        {'params': [torch_model.wheel_Bf, torch_model.wheel_Br], 'lr': 1e-1, 'weight_decay': 1e-2, 'betas': (0.90, 0.999)},
        {'params': [torch_model.wheel_Cf, torch_model.wheel_Cr], 'lr': 1e-2, 'weight_decay': 1e-1, 'betas': (0.90, 0.999)}
    ])
    w = torch.tensor([0, 0, 0, 0, 1, 90/np.pi])

    # optimizer = torch.optim.AdamW([
    #     {'params': [torch_model.wheel_Dr], 'lr': 1e1, 'weight_decay': 1e-5, 'betas': (0.90, 0.999)},
    #     {'params': [torch_model.wheel_Br], 'lr': 1e-1, 'weight_decay': 1e-2, 'betas': (0.90, 0.999)},
    #     {'params': [torch_model.wheel_Cr], 'lr': 1e-2, 'weight_decay': 1e-1, 'betas': (0.90, 0.999)}
    # ])
    # w = torch.tensor([0, 0, 0, 0, 1, 0])

    # optimizer = torch.optim.AdamW([
    #     {'params': [torch_model.wheel_Df], 'lr': 1e1, 'weight_decay': 1e-3, 'betas': (0.90, 0.999)},
    #     {'params': [torch_model.wheel_Bf], 'lr': 1e-1, 'weight_decay': 1e-3, 'betas': (0.90, 0.999)},
    #     {'params': [torch_model.wheel_Cf], 'lr': 1e-2, 'weight_decay': 1e-2, 'betas': (0.90, 0.999)}
    # ])
    # w = torch.tensor([0, 0, 0, 0, 1, 0])
else:
    raise Exception("unknown dataset")

output_folder = "id_results"
os.makedirs(output_folder, exist_ok=True)

epoch_data = train_fn(torch.from_numpy(filtered_inputs), torch.from_numpy(filtered_outputs), torch_model, train_lr=1, epochs=1000, optimizer=optimizer, w=w, device="cuda:0")

filtered_inputs = torch.from_numpy(filtered_inputs).to(torch.float).cuda()
with torch.no_grad():
    model_outputs = torch_model(filtered_inputs).cpu().numpy()
filtered_inputs = filtered_inputs.cpu().numpy()

params = {k: v.item() for k, v in torch_model.state_dict().items()}
if args.type == "acceleration_data":
    with open("dynamics_params/drivetrain_params.json", "w") as outfile:
        outfile.write(json.dumps(params))

    titles = [
        ("vx", 0, True),
        ("throttle", 1, True),
        ("ax", 0, False),
    ]
    filename = "drivetrain_fit.png"
else:
    with open("dynamics_params/bicycle_params.json", "w") as outfile:
        outfile.write(json.dumps(params))

    rates = np.linspace(-1, 1, 100)
    Fry = params["wheel_Dr"] * np.sin(params["wheel_Cr"] * np.arctan(params["wheel_Br"] * rates))
    Ffy = params["wheel_Df"] * np.sin(params["wheel_Cf"] * np.arctan(params["wheel_Bf"] * rates))
    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(rates, Fry)
    ax[0].title.set_text('Rear tire forces')
    ax[1].plot(rates, Ffy)
    ax[1].title.set_text('Front tire forces')
    plt.savefig(output_folder + "/tire_forces.png")

    original_inputs = np.c_[original_inputs, additionals]
    filtered_inputs = np.c_[filtered_inputs, additionals]

    titles = [
        ("hdg", 0, True),
        ("vx", 1, True),
        ("vy", 2, True),
        ("w", 3, True),
        ("steer", 4, True),
        ("throttle", 5, True),
        ("slip", [7, 8, 9], True),
        ("ax", 3, False),
        ("ay", 4, False),
        ("dw", 5, False)
    ]
    filename = "bicycle_fit.png"

fig, ax = plt.subplots(len(titles), 1, figsize=(int(max(len(original_inputs) / 10000, 1) * 30), 40))
fig.tight_layout()

for j, (title, idx, is_input) in enumerate(titles):
    ax[j].set_title(title)

    scaler = 1
    if title in ['hdg', 'w', 'dw']:
        scaler = 180 / np.pi

    if is_input:
        if isinstance(idx, list):
            ax[j].set_ylim(-0.5, 0.5)
            for i in idx:
                ax[j].plot(scaler * original_inputs[:, i], label="raw data")
            ax[j].plot(np.zeros_like(original_inputs[:, 0]))
        else:
            ax[j].plot(scaler * original_inputs[:, idx], label="raw data")
            ax[j].plot(scaler * filtered_inputs[:, idx], label="filtered (training) data")
    else:
        ax[j].plot(scaler * original_outputs[:, idx], label="raw data")
        ax[j].plot(scaler * filtered_outputs[:, idx], label="filtered (training) data")
        ax[j].plot(scaler * model_outputs[:, idx], label="model output")

    ax[j].legend()

plt.savefig(output_folder + "/" + filename)
plt.close()

# fig, ax = plt.subplots(2, 1, figsize=(8, 8))
# ax[0].plot(np.arange(len(epoch_data)), np.log(epoch_data[:, 0]))
# ax[0].title.set_text('train log loss')
# ax[1].plot(np.arange(len(epoch_data)), np.log(epoch_data[:, 1]))
# ax[1].title.set_text('valid log loss')
# if args.type == "acceleration_data":
#     plt.savefig(output_folder + "/drivetrain_loss.png")
# else:
#     plt.savefig(output_folder + "/bicycle_loss.png")
# plt.close()
