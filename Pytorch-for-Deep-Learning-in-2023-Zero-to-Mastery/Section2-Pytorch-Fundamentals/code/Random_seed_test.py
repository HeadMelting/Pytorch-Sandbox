# Let's make some random but reproducible tensors
import torch

# Set the random seed
RANDOM_SEED = 77
torch.manual_seed(RANDOM_SEED)  #
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)

print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)
