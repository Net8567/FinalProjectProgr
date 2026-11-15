import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
x = torch.randn(1000, 1000, device="cuda")
print((x @ x).sum().item())