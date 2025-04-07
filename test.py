from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtGui import QPixmap

# Otwórz obraz i skonwertuj na RGB
image = Image.open("C:/tmp/1.jpg").convert("RGB")

# Konwersja na ImageQt
qt_image = ImageQt(image)
print(type(qt_image)) 
# Bezpośrednia konwersja na QPixmap
pixmap = QPixmap.fromImage(qt_image)  # ImageQt jest już kompatybilny

print("Działa!")