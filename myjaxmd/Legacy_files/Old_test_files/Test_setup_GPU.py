from jax.lib import xla_bridge
print('Jax Device: ', xla_bridge.get_backend().platform)
import torch
x = torch.rand(5, 3).cuda()
print(x)

print('PyTorch Cuda version: ', torch.version.cuda)
print('PyTorch can access GPU: ', torch.cuda.is_available())
print('PyTorch current GPU: ', torch.cuda.current_device())