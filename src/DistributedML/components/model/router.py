import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, num_experts, expert_class, expert_args=()):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([expert_class(*expert_args) for _ in range(num_experts)])
        self.fake_gate_weights = nn.Parameter(torch.ones(num_experts) / num_experts, requires_grad=False)

    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=0)
        output = expert_outputs.mean(dim=0) 
        return output
