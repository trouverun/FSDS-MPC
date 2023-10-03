import config
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler


class DynamicsDataset(Dataset):
    def __init__(self, x, y):
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]


def train_fn(inputs, targets, torch_model, train_lr, epochs, optimizer=None, w=None, device="cuda:0"):
    dataset = DynamicsDataset(inputs, targets)
    all_indices = np.arange(len(inputs))
    train_indices = np.random.choice(all_indices, int(0.8 * len(inputs)), False)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_indices = np.setdiff1d(all_indices, train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=5*1024)
    valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=5*1024)

    if optimizer is None:
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=train_lr, weight_decay=1e-5, betas=(0.90, 0.999))
    loss_fn = torch.nn.MSELoss(reduction="mean")

    if w is None:
        w = torch.ones(targets.shape[1]).to(device)
    else:
        w = w.cuda()

    epoch_data = []
    no_improvement = 0
    best = torch.inf
    for epoch in range(epochs):
        torch_model.train()
        train_loss = 0
        for data in train_loader:
            inputs, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = torch_model(inputs)
            loss = loss_fn(w*outputs, w*labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        torch_model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data[0].to(device, dtype=torch.float), data[1].to(device, dtype=torch.float)
                outputs = torch_model(inputs)
                loss = loss_fn(w*outputs, w*labels)
                val_loss += loss.item()

        epoch_data.append([train_loss, val_loss])
        print("Epoch %3d: Train loss %.8f, Validation loss %.8f" % (epoch, train_loss, val_loss))

        if val_loss < best:
            best = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > config.patience:
                return np.asarray(epoch_data)

    return np.asarray(epoch_data)
