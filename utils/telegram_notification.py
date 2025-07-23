import requests
from utils.token import TOKEN as tk
CHAT_ID_VINCENZO = "521260346"
CHAT_ID_RENATO = "407888332"
def send_telegram_notification(message, CHAT_ID: list):
    url = f"https://api.telegram.org/bot{tk}/sendMessage"
    for id in CHAT_ID:
        payload = {
            "chat_id": id,
            "text": f"{message}",
            "parse_mode": "HTML"  
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            print("Notifica inviata con successo!")
        except requests.exceptions.RequestException as e:
            print(f"Errore nell'invio: {e}")
