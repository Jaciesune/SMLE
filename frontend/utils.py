import requests

def load_stylesheet(filename):
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Plik stylów {filename} nie został znaleziony.")
        return ""

def verify_credentials(username, password):
    try:
        response = requests.post("http://localhost:8000/login", json={"username": username, "password": password})
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