import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
import skimage
import albumentations as A
import torch_directml

# Wybór urządzenia (DirectML lub CUDA lub CPU)
if torch_directml.is_available():
    device = torch_directml.device()
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Informacje o wersjach bibliotek
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("OpenCV version:", cv2.__version__)
print("Matplotlib version:", plt.matplotlib.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-image version:", skimage.__version__)
print("Albumentations version:", A.__version__)
print(f"Używane urządzenie: {device}")
print("Liczba wątków PyTorch:", torch.get_num_threads())
print("Liczba wątków interop PyTorch:", torch.get_num_interop_threads())