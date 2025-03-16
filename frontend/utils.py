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
        # Wysyłamy zapytanie do backendu
        response = requests.post("http://localhost:8000/login", json={"username": username, "password": password})

        print(f"Odpowiedź z backendu: {response.text}")  # Logowanie odpowiedzi z serwera

        if response.status_code == 200:
            # Sprawdzamy, czy odpowiedź zawiera dane roli
            data = response.json()

            if "role" in data and "role" in data["role"]:  # Upewniamy się, że "role" jest w odpowiedzi
                return data["role"]["role"]  # Zwracamy rolę użytkownika
            else:
                print("Brak roli w odpowiedzi serwera.")
                return None
        else:
            print(f"Niepoprawne dane logowania: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Błąd połączenia z serwerem: {e}")
        return None