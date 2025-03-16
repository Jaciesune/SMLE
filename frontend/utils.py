import requests

def load_stylesheet(filename):
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Plik stylów {filename} nie został znaleziony.")
        return ""

def verify_credentials(username, password):
    # Wysyłamy zapytanie HTTP do backendu w formacie JSON
    try:
        response = requests.post("http://localhost:8000/login", json={"username": username, "password": password})

        if response.status_code == 200:
            # Jeśli odpowiedź jest pozytywna, zwróć rolę użytkownika
            return response.json().get("role")
        else:
            # Jeśli logowanie się nie udało, zwróć None
            return None
    except requests.exceptions.RequestException as e:
        print(f"Błąd połączenia z serwerem: {e}")
        return None