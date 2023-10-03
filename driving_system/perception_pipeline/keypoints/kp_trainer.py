import torch
import numpy as np
import argparse
from torch.utils.data import SubsetRandomSampler, DataLoader
from driving_system.perception_pipeline.keypoints.kp_model import KeypointModel
from driving_system.perception_pipeline.keypoints.kp_dataset import KeypointDataset


def delta(pi, pj):
    return torch.sqrt(torch.sum(torch.square(pi - pj)))


def cr(p1, p2, p3, p4):
    return (delta(p1, p3) / delta(p1, p4)) / (delta(p2, p3) / delta(p2, p4))


def keypoint_loss(out, label, l=0.0001, cr3d=1.8333):
    ps = out.reshape(out.shape[0], 7, 2)
    p1 = ps[:, 0, :]
    p2 = ps[:, 1, :]
    p3 = ps[:, 2, :]
    p4 = ps[:, 3, :]
    p5 = ps[:, 4, :]
    p6 = ps[:, 5, :]
    p7 = ps[:, 6, :]
    return (torch.nn.MSELoss()(out, label) +
            (l*torch.square(cr(p1, p2, p3, p4) - cr3d) + l*torch.square(cr(p1, p5, p6, p7) - cr3d)) / out.shape[0])


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-l', '--learning_rate', type=float, default=1e-2)
parser.add_argument('-w', '--weight_decay', type=float, default=1e-4)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-p', '--patience', type=int, default=20)
parser.add_argument('-d', '--dataset_path', type=str)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_model = KeypointModel().to(device)
initial_weights = 'driver/vision_pipeline/keypoints/model.pt'
torch.save(torch_model.model.state_dict(), initial_weights)

kp_dataset = KeypointDataset('files.txt', args.dataset_path, transforms=True)
all_indices = np.arange(len(kp_dataset))
train_indices = np.random.choice(all_indices, int(0.75 * len(kp_dataset)), False)
train_sampler = SubsetRandomSampler(train_indices)
valid_indices = np.setdiff1d(all_indices, train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
train_loader = DataLoader(kp_dataset, sampler=train_sampler, batch_size=args.batch_size)
valid_loader = DataLoader(kp_dataset, sampler=valid_sampler, batch_size=args.batch_size)

optimizer = torch.optim.AdamW(torch_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.90, 0.999))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer, max_lr=args.lr, total_steps=None,
    epochs=args.epochs, steps_per_epoch=len(train_loader), pct_start=0.3,
    anneal_strategy='cos', cycle_momentum=True,
    base_momentum=0.85, max_momentum=0.95, div_factor=25.0,
    final_div_factor=1000.0, last_epoch=-1
)

for epoch in range(args.epoch):
    torch_model.train()
    train_loss = 0
    train_truths = torch.tensor([]).to(device)
    train_predicts = torch.tensor([]).to(device)
    for data in train_loader:
        inputs, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
        optimizer.zero_grad()
        outputs = torch_model(inputs)
        loss = keypoint_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        with torch.no_grad():
            train_truths = torch.cat((train_truths, labels), dim=0)
            train_predicts = torch.cat((train_predicts, outputs), dim=0)
        scheduler.step()

    torch_model.eval()
    val_loss = 0
    val_truths = torch.tensor([]).to(device)
    val_predicts = torch.tensor([]).to(device)
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            outputs = torch_model(inputs)
            loss = keypoint_loss(outputs, labels)
            val_loss += loss.item()
            val_truths = torch.cat((val_truths, labels), dim=0)
            val_predicts = torch.cat((val_predicts, outputs), dim=0)

    print("Epoch %d: Train loss %.8f, Validation loss %.8f" % (epoch, train_loss, val_loss))