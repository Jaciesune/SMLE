# SMLE (System Maszynowego Liczenia Elementów)

## Opis projektu

**SMLE (System Maszynowego Liczenia Elementów)** to zaawansowana aplikacja do automatycznego zliczania i analizy elementów na obrazach przy użyciu technik widzenia komputerowego i uczenia maszynowego. Projekt integruje backend oparty na modelach głębokiego uczenia (Mask R-CNN, Faster R-CNN, MRCNN) z intuicyjnym interfejsem użytkownika, umożliwiającym zliczanie obiektów, trening modeli, oznaczanie danych, zarządzanie zbiorami danych, benchmarking oraz administrację użytkownikami. Wykorzystuje technologie takie jak Python, PyTorch, OpenCV, Docker i Docker Compose, zapewniając skalowalność i łatwość wdrożenia. Aplikacja jest przeznaczona do zastosowań w przemyśle, logistyce czy badaniach naukowych, gdzie wymagana jest precyzyjna detekcja i analiza obiektów na obrazach.

## Uruchomienie projektu

1. Upewnij się, że masz zainstalowane **Docker** oraz **Docker Compose**.

2. Uruchom projekt za pomocą:

   ```bash
   docker-compose up --build
   ```

3. Frontend uruchamiany jest ręcznie:

   - Zainstaluj wymagane biblioteki z pliku `frontend/requirements.txt`:

     ```bash
     pip install -r frontend/requirements.txt
     ```

   - Uruchom aplikację frontendową:

     ```bash
     python frontend/main.py
     ```

## Najważniejsze technologie użyte w projekcie

- **Python**: Główny język programowania dla backendu i frontendu.
- **Docker**: Konteneryzacja aplikacji dla łatwego wdrażania.
- **Docker Compose**: Orkiestracja wielu kontenerów.
- **Mask R-CNN**: Model głębokiego uczenia do segmentacji instancji.
- **Faster R-CNN**: Model do wykrywania obiektów.
- **MRCNN**: Rozszerzenie Mask R-CNN dla zaawansowanej detekcji.
- **PyTorch**: Framework do uczenia maszynowego.
- **OpenCV**: Biblioteka do przetwarzania obrazów.

# Interfejs użytkownika

## 1. Zliczanie
- \
  Ekran zliczania umożliwia wgranie obrazu wejściowego do analizy. Użytkownik może wybrać modele Mask R-CNN, Faster R-CNN lub MCNN. Możliwa jest również opcja wyboru preprocesingu który znacznie poprawia wyniki detekcji MCNN.
- \
  Wynik detekcji obiektów na obrazie. Pokazuje obraz z naniesionymi maskami lub ramkami bounding box wokół wykrytych elementów, wraz z ich liczbą i etykietami klas, wygenerowanymi przez wybrany model.
- \
  Poniżej przedstawiono przykład detekcji przy użyciu modelu Mask R-CNN.
![](docs/screenshots/detekcja_przed.png)
![](docs/screenshots/detekcja_po.png)

## 2. Trening
- \
  Moduł treningu pozwala użytkownikowi na konfigurację i uruchamianie procesu uczenia modeli głębokiego uczenia. Umożliwia wybór hiperparametrów, takich jak liczba epok, learning rate, oraz wgranie oznaczonych zbiorów danych do treningu.
![](docs/screenshots/trening.png)

## 3. Modele
Sekcja zarządzania modelami wyświetla listę dostępnych modeli (np. Mask R-CNN, Faster R-CNN, MCNN) wraz z ich metadanymi, takimi jak data treningu, wersja czy metryki wydajności.
![](docs/screenshots/modele.png)

## 4. Oznaczanie zdjęć
- \
  Narzędzie do półautomatycznego oznaczania obrazów umożliwia użytkownikowi dodawanie etykiet i masek dla obiektów na zdjęciach, tworząc dane treningowe dla modeli. Interfejs wspiera precyzyjne rysowanie poligonów i bounding boxów.
- \
  Narzędzie pozwala na wybór wcześniej wytrenowanego modelu Mask R-CNN, którego zadaniem jest wstępne oznaczenie zdjęć co przyspiesza proces oznaczania.
![](docs/screenshots/oznaczanie.png)
- \
  Widok skrótów klawiszowych dla narzędzia do oznaczania, ułatwiających szybkie wykonywanie operacji, takich jak przełączanie klas, zapisywanie etykiet czy nawigacja między obrazami.
![](docs/screenshots/oznaczanie_skroty.png)

## 5. Tworzenie zbiorów danych
- \
  Moduł tworzenia zbiorów danych pozwala na organizację i walidację oznaczonych obrazów w zestawy treningowe, walidacyjne i testowe. Użytkownik może zarządzać metadanymi zbiorów i eksportować je w formatach zgodnych z PyTorch.
![](docs/screenshots/zbior_danych.png)

## 6. Benchmark

- \
  Ekran benchmarku prezentuje wyniki testów wydajności modeli na wybranym zbiorze danych, w tym metryki takie jak MAE oraz skuteczność.
- \
  Porównanie wydajności różnych modeli (np. Mask R-CNN vs. Faster R-CNN) w formie listy z podziałem na zestaw danch, umożliwiające wybór optymalnego modelu dla konkretnego zastosowania oraz najlepszego modelu z całego zestawienia.
![](docs/screenshots/benchmark.png)
![](docs/screenshots/benchmark_porownanie.png)

## 7. Użytkownicy
- \
  Panel administracyjny do zarządzania użytkownikami aplikacji. Pozwala na dodawanie i podgląd kont.
![](docs/screenshots/uzytkownicy.png)



# Licencje

Projekt korzysta z bibliotek open source. Pełna lista bibliotek, ich wersje, licencje oraz data weryfikacji znajdują się w pliku `licences/THIRD_PARTY_LICENSES.md`. Teksty licencji dla każdej biblioteki są dostępne w folderze `licences`.