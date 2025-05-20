import torch
import os
import wandb
import torch
import numpy as np
import torch.nn as nn

from DistributedML.entity import ModelTrainerConfig
from DistributedML.components.model.expert import UNet3DExpert
from DistributedML.components.model.router import MoELayer
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

class DatasetLoader(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def generate_synthetic_data(num_samples=20, shape=(64, 64, 32)):
    images = np.random.rand(num_samples, *shape).astype(np.float32)
    labels = np.random.randint(0, 2, size=(num_samples, *shape)).astype(np.int64)
    return images, labels

def get_dataloaders(batch_size=4):
    train_images, train_labels = generate_synthetic_data(20)
    val_images, val_labels = generate_synthetic_data(5)

    train_dataset = DatasetLoader(train_images, train_labels)
    val_dataset = DatasetLoader(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        wandb.init(project="brain-tumor-moe", mode="online")

    
    def train(rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
        train_loader, _ = get_dataloaders()

        model = MoELayer(
            num_experts=4,
            input_dim=128 * 128 * 64,
            expert_class=UNet3DExpert,
            expert_args=(1, 2)
        )
        
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        model = DDP(model, device_ids=[rank])

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(10):
            model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                wandb.log({"epoch": epoch + 1, "batch": batch_idx, "loss": loss.item()})
                print(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")
        
        if rank == 0:
            print("Training completed on rank 0.")
            wandb.finish()
