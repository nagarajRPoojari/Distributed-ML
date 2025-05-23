# import torch
# from torchviz import make_dot

# # Assuming UNet3DExpert and MoELayer classes are defined
# from src.DistributedML.components.model.expert import UNet3DExpert
# from src.DistributedML.components.model.router import MoELayer
# # Instantiate the model
# moe_unet = MoELayer(num_experts=3, expert_class=UNet3DExpert, expert_args=(1, 2))

# # Dummy input (e.g., [batch, channels, depth, height, width])
# x = torch.randn(1, 1, 64, 64, 64)

# # Get the output
# output = moe_unet(x)

# # Generate diagram1
# make_dot(output, params=dict(moe_unet.named_parameters())).render("moe_unet", format="png")
import torch
import torch.nn as nn
from torchviz import make_dot

# Minimal expert: just a single Conv3D layer
class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 2, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# Simple MoE layer with 4 experts
class SimpleMoE(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([TinyExpert() for _ in range(num_experts)])

    def forward(self, x):
        # Average outputs from all experts (no gating logic here)
        outputs = torch.stack([expert(x) for expert in self.experts])
        return outputs.mean(0)

# Instantiate and visualize
model = SimpleMoE(num_experts=4)
x = torch.randn(1, 1, 8, 8, 8)  # tiny 3D volume
y = model(x)

# Generate and save the computation graph
make_dot(y, params=dict(model.named_parameters())).render("super_simple_moe", format="png")
