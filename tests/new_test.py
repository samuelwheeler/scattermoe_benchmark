import torch

x = torch.randn((100, 1000), device = 'xpu:0')

print(x.device)