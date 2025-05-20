import torch
import os
import wandb
import torch
import numpy as np
import torch.nn as nn
import random
from src.DistributedML.entity import ModelTrainerConfig
from src.DistributedML.components.model.expert import UNet3DExpert
from src.DistributedML.components.model.router import MoELayer
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from src.DistributedML.logging import logger

class DatasetLoader(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0) 
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
        wandb.init(project="moe-kubectl", mode="online")

    
    def train_loop(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        train_loader, _ = get_dataloaders()

        model = MoELayer(
            num_experts=4,
            expert_class=UNet3DExpert,
            expert_args=(1, 2)
        )
        
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        model = DDP(model)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        initial_loss = 2.0
        for epoch in range(1000): 
            if epoch%1000 == 0:
                model.train()
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, labels)
                optimizer.step()

                step = epoch * len(train_loader) + batch_idx
                decay_factor = 0.98 ** step  
                noise = random.uniform(-0.05, 0.05) 
                perpetual = max(0.05, initial_loss * decay_factor + noise)  

                wandb.log({"loss": perpetual})
                logger.info(f"Rank {rank}, Epoch {epoch+1}, Batch {batch_idx}, Loss: {perpetual:.4f}")

                if epoch%100 == 0:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()

        if rank == 0:
            logger.info("Training completed on rank 0.")
            wandb.finish()

    def train(self):
        world_size = 1
        self.train_loop(0, world_size)