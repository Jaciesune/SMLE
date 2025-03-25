import torch

# Ustawienie urządzenia na GPU (CUDA), jeśli dostępne
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tworzenie tensora na wybranym urządzeniu
x = torch.tensor([123.0], device=device)

print("Tensor utworzony na:", x.device)
