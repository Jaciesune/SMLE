# 🔧 DirectML Test Pack

## Zawartość:
- `gpuTest.py` – test tworzenia tensora na DirectML
- `DirectML.dll` – biblioteka do załadowania ręcznego (1.11.0)
- `requirements.txt` – zależności
- `README.txt` – to co czytasz :)

## Jak używać:
1. Utwórz środowisko:
   python -m venv dml_venv

2. Aktywuj:
   .\dml_venv\Scripts\activate

3. Zainstaluj zależności:
   pip install -r requirements.txt --no-deps

4. Uruchom test:
   python gpuTest.py

Tensor powinien zostać utworzony na urządzeniu `privateuseone:0`.
