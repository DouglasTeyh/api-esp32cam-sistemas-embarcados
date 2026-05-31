import os
import cv2
import numpy as np
import json
import threading
import time
import requests
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

# Limitar recursos para ambiente de nuvem
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
    net = cv2.dnn.readNetFromONNX(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None
    if net:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Warm-up inference to preload libraries and speed up the first real request
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(dummy_img, 1/255.0, (512, 512), swapRB=True, crop=False)
        net.setInput(blob)
        net.forward()
        print("Modelo YOLO carregado com sucesso via OpenCV DNN.")
except Exception as e:
    print(f"Erro ao carregar YOLO via OpenCV DNN: {e}")
    net = None

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
    # Obter o último update_id para iniciar o polling sem reprocessar mensagens antigas
    try:
        res = requests.get(f"{url}getUpdates", params={"limit": 1}, timeout=10)
        if res.status_code == 200:
            results = res.json().get("result", [])
            if results:
                last_id = results[0]["update_id"]
    except Exception as e:
        print(f"Erro inicial de getUpdates: {e}")

    while True:
        try:
            res = requests.get(f"{url}getUpdates", params={"offset": last_id + 1, "timeout": 20}, timeout=25)
            if res.status_code == 200:
                for update in res.json().get("result", []):
                    last_id = update["update_id"]
                    
                    # Processa cada update de forma independente para evitar travar o loop
                    try:
                        msg = update.get("message")
                        if not msg:
                            continue
                        
                        chat = msg.get("chat")
                        if not chat:
                            continue
                            
                        chat_id = chat.get("id")
                        text = msg.get("text") or ""
                        
                        if text in ("/start", "➕ Registrar Novo"):
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": "Bem-vindo ao *VenomESP*! Envie o número de série do dispositivo (ESP32-CAM-XX:XX:XX:XX:XX:XX).", "parse_mode": "Markdown"},
                                timeout=10
                            )
                        elif text == "❓ Ajuda":
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": "🦂 *VenomESP*\nMonitora: Escorpiões, Cobras, Aranhas e Centopeias.", "parse_mode": "Markdown"},
                                timeout=10
                            )
                        elif text.startswith("ESP32-CAM-"):
                            with registrations_lock: 
                                registrations[text] = chat_id
                            save_registrations()
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": "✅ Dispositivo registrado!"},
                                timeout=10
                            )
                    except Exception as inner_e:
                        print(f"Erro ao processar update {last_id}: {inner_e}")
            else:
                print(f"getUpdates retornou HTTP {res.status_code}")
                time.sleep(5)
        except Exception as e:
            print(f"Erro no loop de polling do Telegram: {e}")
            time.sleep(5)

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

CLASS_NAMES_PT = {
    0: 'Centopeia',
    1: 'Escorpião',
    2: 'Cobra',
    3: 'Aranha'
}

def predict_yolo(img):
    h_orig, w_orig = img.shape[:2]
    
    # Preprocess
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward() # shape [1, 8, 5376]
    
    output = outputs[0].T # shape [5376, 8]
    
    boxes = []
    confidences = []
    class_ids = []
    
    for i in range(output.shape[0]):
        row = output[i]
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        confidence = classes_scores[class_id]
        
        if confidence >= 0.25:
            x_center, y_center, w, h = row[0:4]
            x = int((x_center - w / 2) * (w_orig / 512.0))
            y = int((y_center - h / 2) * (h_orig / 512.0))
            width = int(w * (w_orig / 512.0))
            height = int(h * (h_orig / 512.0))
            
            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
            
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    
    detections = []
    img_draw = img.copy()
    
    colors = {
        0: (0, 140, 255),    # Laranja
        1: (0, 0, 255),      # Vermelho
        2: (0, 255, 0),      # Verde
        3: (255, 0, 128)     # Roxo
    }
    
    if len(indices) > 0:
        flat_indices = np.array(indices).flatten()
        for idx in flat_indices:
            x, y, w, h = boxes[idx]
            class_id = class_ids[idx]
            label = CLASS_NAMES_PT.get(class_id, "Desconhecido")
            score = confidences[idx]
            
            detections.append(label)
            
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_orig - x)
            h = min(h, h_orig - y)
            
            color = colors.get(class_id, (0, 255, 255))
            
            # Desenha retângulo elegante
            cv2.rectangle(img_draw, (x, y), (x + w, y + h), color, 2)
            
            # Badge para a etiqueta
            label_text = f"{label} ({int(score * 100)}%)"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img_draw, (x, y - text_h - 8), (x + text_w + 10, y), color, -1)
            cv2.putText(img_draw, label_text, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            
    return detections, img_draw

@app.post("/detectar", response_model=DeteccaoResponse)
def detectar_animal(background_tasks: BackgroundTasks, file: UploadFile = File(...), dispositivo_id: str = Form(...)):
    if net is None:
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Modelo YOLO não carregado.")
    try:
        img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Falha ao decodificar imagem.")
            
        detections, img_draw = predict_yolo(img)
        
        if len(detections) > 0:
            animais = ", ".join(set(detections))
            dispositivo_limpo = dispositivo_id.replace(":", "_")
            path = f"static/alerta_{dispositivo_limpo}_{int(time.time())}.jpg"
            cv2.imwrite(path, img_draw)
            
            with registrations_lock: chat_id = registrations.get(dispositivo_id)
            if chat_id: background_tasks.add_task(send_telegram_alert, path, chat_id, dispositivo_id, animais)
            
            return DeteccaoResponse(animal_detectado=True, acionar_alarme=True)
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False)
    except Exception as e: return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)