import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse


# Przykładowa baza danych użytkowników (może być w pliku JSON lub SQL, ale tu zrobimy ją prostą)
users = {
    "user1": {"password": "password123", "role": "admin"},
    "user2": {"password": "password456", "role": "user"},
}

# Klasa obsługująca zapytania HTTP
class RequestHandler(BaseHTTPRequestHandler):
    # Metoda obsługująca żądanie POST
    def do_POST(self):
        # Parsowanie ścieżki URL
        path = self.path
        if path == "/login":
            # Parsowanie danych JSON w zapytaniu POST
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))
            
            username = data.get('username')
            password = data.get('password')
            
            # Logika logowania
            if username in users and users[username]["password"] == password:
                response = {
                    "token": "example_token",  # Możesz tutaj wygenerować prawdziwy token
                    "role": users[username]["role"]
                }
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            else:
                response = {"error": "Invalid credentials"}
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
        else:
            # Obsługuje inne zapytania (np. GET), może odpowiedzieć 404
            self.send_response(404)
            self.end_headers()

# Funkcja uruchamiająca serwer
def run(server_class=HTTPServer, handler_class=RequestHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

# Uruchomienie serwera
if __name__ == '__main__':
    run()
