import requests

def load_stylesheet(filename):
    try:
        # Tworzymy pełną ścieżkę do pliku w katalogu frontend/styles/
        full_path = f"frontend/styles/{filename}"
        with open(full_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Plik stylów {full_path} nie został znaleziony.")
        return ""
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku stylów {full_path}: {e}")
        return ""

def verify_credentials(username, password, api_url):
    try:

        response = requests.post(f"{api_url}/login", json={"username": username, "password": password})
        print(f"Odpowiedź z backendu: {response.text}")  # Debugowanie

        if response.status_code == 200:
            data = response.json()
            print(f"Odpowiedź JSON: {data}")  # Sprawdzenie struktury odpowiedzi
            
            if "role" in data:
                return data["role"]  # Pobieramy rolę bez dodatkowych poziomów
            else:
                print("Brak pola 'role' w odpowiedzi serwera.")
                return None
        else:
            print(f"Niepoprawne dane logowania: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Błąd połączenia z serwerem: {e}")
        return None