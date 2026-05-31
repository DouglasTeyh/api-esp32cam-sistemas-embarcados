import os
import cv2
import numpy as np
import json
import threading
import time
import requests
import torch
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Limitar recursos para ambiente de nuvem
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

load_dotenv()

app = FastAPI(
    title="API Detecção de Animais Peçonhentos - VenomESP",
    description="API para detecção de escorpiões, cobras, aranhas e centopeias via ESP32-CAM.",
    version="2.1.0"
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

YOLO_MODEL_PATH = "best.onnx"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("API_URL", "http://localhost:8000")

try:
    model = YOLO(YOLO_MODEL_PATH, task="detect") if os.path.exists(YOLO_MODEL_PATH) else None
except Exception as e:
    print(f"Erro ao carregar YOLO: {e}")
    model = None

REGISTRATIONS_FILE = "registrations.json"
registrations = {}
registrations_lock = threading.Lock()
bot_username = "BotDesconhecido"

def load_registrations():
    global registrations
    if os.path.exists(REGISTRATIONS_FILE):
        try:
            with open(REGISTRATIONS_FILE, "r") as f:
                registrations = json.load(f)
        except: registrations = {}
    else: registrations = {}

def save_registrations():
    with open(REGISTRATIONS_FILE, "w") as f:
        with registrations_lock:
            json.dump(registrations, f, indent=4)

load_registrations()

def send_telegram_alert(image_path: str, chat_id: int, dispositivo_id: str, animais: str):
    if not TELEGRAM_BOT_TOKEN: return
    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    caption = f"🚨 *ALERTA DE SEGURANÇA* 🚨\n\nDetectado(s): *{animais}*\nDispositivo: `{dispositivo_id}`"
    try:
        with open(image_path, "rb") as photo_file:
            requests.post(url_photo, data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}, files={"photo": photo_file}, timeout=20)
    except Exception as e: print(f"Erro Telegram: {e}")

def telegram_bot_polling():
    global bot_username
    if not TELEGRAM_BOT_TOKEN: return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    
    # Obter nome do bot para a resposta da API config
    try:
        res = requests.get(f"{url}getMe", timeout=10)
        if res.status_code == 200:
            bot_username = res.json().get("result", {}).get("username", "BotDesconhecido")
            print(f"Bot carregado: @{bot_username}")
    except Exception as e:
        print(f"Erro ao consultar getMe no Telegram: {e}")

    last_id = 0
    while True:
        try:
            res = requests.get(f"{url}getUpdates", params={"offset": last_id + 1, "timeout": 20}, timeout=25)
            if res.status_code == 200:
                for update in res.json().get("result", []):
                    last_id = update["update_id"]
                    msg = update.get("message", {})
                    chat_id = msg.get("chat", {}).get("id")
                    text = msg.get("text") or ""
                    
                    if text in ("/start", "➕ Registrar Novo"):
                        requests.post(f"{url}sendMessage", json={"chat_id": chat_id, "text": "Bem-vindo ao *VenomESP*! Envie o número de série do dispositivo (ESP32-CAM-XX:XX:XX:XX:XX:XX).", "parse_mode": "Markdown"})
                    elif text == "❓ Ajuda":
                        requests.post(f"{url}sendMessage", json={"chat_id": chat_id, "text": "🦂 *VenomESP*\nMonitora: Escorpiões, Cobras, Aranhas e Centopeias.", "parse_mode": "Markdown"})
                    elif text.startswith("ESP32-CAM-"):
                        with registrations_lock: registrations[text] = chat_id
                        save_registrations()
                        requests.post(f"{url}sendMessage", json={"chat_id": chat_id, "text": "✅ Dispositivo registrado!"})
        except: time.sleep(5)

threading.Thread(target=telegram_bot_polling, daemon=True).start()

class DeteccaoResponse(BaseModel):
    animal_detectado: bool
    acionar_alarme: bool
    erro: str | None = None

@app.get("/config")
def get_config():
    return {"bot_username": bot_username}

@app.get("/status-dispositivo/{dispositivo_id}")
def check_dispositivo_status(dispositivo_id: str):
    with registrations_lock:
        registrado = dispositivo_id in registrations
    return {"registrado": registrado}

@app.post("/detectar", response_model=DeteccaoResponse)
def detectar_animal(background_tasks: BackgroundTasks, file: UploadFile = File(...), dispositivo_id: str = Form(...)):
    if model is None:
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Modelo YOLO não carregado.")
    try:
        img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Otimização: conf=0.25 (Recall) e imgsz=512 (Resolução nativa do modelo), device="cpu" para robustez
        results = model.predict(source=img, conf=0.25, imgsz=512, save=False, device="cpu")[0]
        
        if len(results.boxes) > 0:
            animais = ", ".join(set([model.names[int(b.cls)] for b in results.boxes]))
            dispositivo_limpo = dispositivo_id.replace(":", "_")
            path = f"static/alerta_{dispositivo_limpo}_{int(time.time())}.jpg"
            cv2.imwrite(path, results.plot())
            
            with registrations_lock: chat_id = registrations.get(dispositivo_id)
            if chat_id: background_tasks.add_task(send_telegram_alert, path, chat_id, dispositivo_id, animais)
            
            return DeteccaoResponse(animal_detectado=True, acionar_alarme=True)
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False)
    except Exception as e: return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)