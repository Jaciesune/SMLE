import torch
import torch_directml

print("Dostępne urządzenia DirectML:")
for i in range(torch_directml.device_count()):
    print(f" - Urządzenie {i}: {torch_directml.device(i)}")

device = torch_directml.device()
print(f"Używane urządzenie: {device}")

x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print("Tensor na DirectML:", x)