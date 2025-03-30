import torch
import torch_directml

# Wymuszenie użycia DirectML
dml_device = torch_directml.device()

# Tworzenie tensora i przesłanie go na DirectML
tensor = torch.ones(3, 3).to(dml_device)
print("✅ Tensor na DirectML:", tensor.device)