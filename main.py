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
                        text = (msg.get("text") or "").strip()
                        
                        # Menu Keyboard
                        menu_keyboard = {
                            "keyboard": [
                                [{"text": "📋 Meus Dispositivos"}, {"text": "➕ Registrar Novo"}],
                                [{"text": "❓ Ajuda"}]
                            ],
                            "resize_keyboard": True
                        }
                        
                        if text in ("/start", "➕ Registrar Novo"):
                            welcome_msg = (
                                "Olá! Bem-vindo ao *Bot VenomESP* 🦂🐍🕷️\n\n"
                                "Para receber alertas de animais peçonhentos do seu ESP32-CAM, "
                                "por favor envie o *Número de Série* do seu dispositivo.\n\n"
                                "Exemplo:\n`ESP32-CAM-XX:XX:XX:XX:XX:XX`"
                            )
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": welcome_msg, "parse_mode": "Markdown", "reply_markup": menu_keyboard},
                                timeout=10
                            )
                        elif text == "📋 Meus Dispositivos":
                            with registrations_lock:
                                user_devices = [dev for dev, cid in registrations.items() if cid == chat_id]
                            if user_devices:
                                devices_list = "\n".join(f"• `{dev}`" for dev in user_devices)
                                msg_text = f"📋 *Seus Dispositivos Cadastrados:*\n\n{devices_list}\n\nVocê receberá alertas em tempo real para todos eles!"
                            else:
                                msg_text = "Você ainda não possui nenhum dispositivo VenomESP cadastrado.\nEnvie o número de série para cadastrar!"
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": msg_text, "parse_mode": "Markdown", "reply_markup": menu_keyboard},
                                timeout=10
                            )
                        elif text == "❓ Ajuda":
                            help_msg = (
                                "🦂 *Ajuda - Sistema VenomESP*\n\n"
                                "Este bot recebe alertas de imagens do seu ESP32-CAM sempre que um escorpião, cobra, aranha ou centopeia for detectado pela nossa inteligência artificial.\n\n"
                                "• Para cadastrar um dispositivo, basta enviar o número de série completo (ex: `ESP32-CAM-XX:XX:XX:XX:XX:XX`).\n"
                                "• Você pode cadastrar múltiplos dispositivos para este mesmo chat.\n"
                                "• Quando o ESP32-CAM inicializar, ele enviará uma foto de teste para confirmar o funcionamento.\n\n"
                                "Use o menu abaixo para navegar!"
                            )
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": help_msg, "parse_mode": "Markdown", "reply_markup": menu_keyboard},
                                timeout=10
                            )
                        elif text.upper().startswith("ESP32-CAM-"):
                            device_id_clean = text.upper()
                            with registrations_lock: 
                                registrations[device_id_clean] = chat_id
                            save_registrations()
                            confirm_msg = (
                                f"✅ *Dispositivo registrado com sucesso!*\n\n"
                                f"Dispositivo ID: `{device_id_clean}`\n"
                                f"Chat ID: `{chat_id}`\n\n"
                                "Você receberá alertas em tempo real sempre que um animal peçonhento for detectado neste dispositivo!"
                            )
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": confirm_msg, "parse_mode": "Markdown", "reply_markup": menu_keyboard},
                                timeout=10
                            )
                        elif text:
                            invalid_msg = (
                                "⚠️ *Comando ou formato inválido.*\n\n"
                                "Para registrar seu dispositivo, envie o número de série no formato:\n"
                                "`ESP32-CAM-XX:XX:XX:XX:XX:XX`"
                            )
                            requests.post(
                                f"{url}sendMessage", 
                                json={"chat_id": chat_id, "text": invalid_msg, "parse_mode": "Markdown", "reply_markup": menu_keyboard},
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
    device_id_clean = dispositivo_id.strip().upper()
    with registrations_lock:
        registrado = device_id_clean in registrations
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
    
    max_conf = 0.0
    max_class_id = -1
    
    for i in range(output.shape[0]):
        row = output[i]
        classes_scores = row[4:]
        class_id = np.argmax(classes_scores)
        confidence = classes_scores[class_id]
        
        if confidence > max_conf:
            max_conf = float(confidence)
            max_class_id = int(class_id)
        
        if confidence >= 0.28:
            x_center, y_center, w, h = row[0:4]
            x = int((x_center - w / 2) * (w_orig / 512.0))
            y = int((y_center - h / 2) * (h_orig / 512.0))
            width = int(w * (w_orig / 512.0))
            height = int(h * (h_orig / 512.0))
            
            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(int(class_id))
            
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.28, 0.45)
    
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
            
    return detections, img_draw, max_conf, max_class_id

@app.post("/detectar", response_model=DeteccaoResponse)
def detectar_animal(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    dispositivo_id: str = Form(...),
    is_test: bool = Form(False)
):
    if net is None:
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Modelo YOLO não carregado.")
    try:
        contents = file.file.read()
        dispositivo_clean = dispositivo_id.strip().upper()
        dispositivo_limpo = dispositivo_id.replace(":", "_")

        if is_test:
            path = f"static/teste_{dispositivo_limpo}_{int(time.time())}.jpg"
            with open(path, "wb") as f:
                f.write(contents)

            with registrations_lock: chat_id = registrations.get(dispositivo_clean)
            if chat_id:
                def send_test_alert(image_path: str, chat_id: int, dispositivo_id: str):
                    if not TELEGRAM_BOT_TOKEN: return
                    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                    caption = (
                        f"📸 *CONEXÃO ESTABELECIDA* 📸\n\n"
                        f"O dispositivo `{dispositivo_id}` foi inicializado com sucesso e enviou esta primeira imagem de teste!\n\n"
                        f"🟢 Status: *Online e Monitorando*"
                    )
                    try:
                        with open(image_path, "rb") as photo_file:
                            requests.post(url_photo, data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}, files={"photo": photo_file}, timeout=20)
                    except Exception as e: print(f"Erro Telegram teste: {e}")

                background_tasks.add_task(send_test_alert, path, chat_id, dispositivo_id)
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False)

        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Falha ao decodificar imagem.")
            
        detections, img_draw, max_conf, max_class_id = predict_yolo(img)
        
        # Logger do terminal/Render de classificação
        max_label = CLASS_NAMES_PT.get(max_class_id, "Nenhuma")
        print(f"[IA-Debug] ID: {dispositivo_id} | Maior Confiança Encontrada: {max_conf*100:.2f}% para Classe '{max_label}'")
        
        if len(detections) > 0:
            animais = ", ".join(set(detections))
            path = f"static/alerta_{dispositivo_limpo}_{int(time.time())}.jpg"
            cv2.imwrite(path, img_draw)
            
            with registrations_lock: chat_id = registrations.get(dispositivo_clean)
            if chat_id: background_tasks.add_task(send_telegram_alert, path, chat_id, dispositivo_id, animais)
            
            return DeteccaoResponse(animal_detectado=True, acionar_alarme=True)
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False)
    except Exception as e: return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)