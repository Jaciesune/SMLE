def load_stylesheet(filename):
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"Plik stylów {filename} nie został znaleziony.")
        return ""

def verify_credentials(username, password):
    # Prosta weryfikacja na potrzeby przykładu.
    users_db = {
        "admin": {"password": "admin", "role": "admin"},
        "user": {"password": "user", "role": "user"},
    }

    if username in users_db and users_db[username]["password"] == password:
        return users_db[username]["role"]
    else:
        return None