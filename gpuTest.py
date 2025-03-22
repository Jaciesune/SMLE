import os
import ctypes

dll_path = os.path.abspath("onnxruntime.dll")
ctypes.windll.LoadLibrary(dll_path)

import torch
import torch_directml

device = torch_directml.device()
x = torch.tensor([123.0], device=device)
print("✅ Tensor utworzony na:", x.device)
