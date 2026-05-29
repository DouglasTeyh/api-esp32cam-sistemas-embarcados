import os
import cv2
import numpy as np
import json
import threading
import time
import requests
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="API Detecção de Animais Peçonhentos - ESP32Cam",
    description="API que recebe imagens de um ESP32, analisa através do YOLOv8 se há Animais Peçonhentos e alerta via Telegram.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Garantir diretório static para imagens salvas
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

YOLO_MODEL_PATH = "best.pt"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Carregar modelo YOLOv8
try:
    if os.path.exists(YOLO_MODEL_PATH):
        model = YOLO(YOLO_MODEL_PATH)
    else:
        model = None
except Exception as e:
    print(f"Erro ao carregar o modelo YOLO: {e}")
    model = None

# Persistência de registros (dispositivo_id -> chat_id)
REGISTRATIONS_FILE = "registrations.json"
registrations = {}
registrations_lock = threading.Lock()

def load_registrations():
    global registrations
    if os.path.exists(REGISTRATIONS_FILE):
        try:
            with open(REGISTRATIONS_FILE, "r") as f:
                with registrations_lock:
                    registrations = json.load(f)
            print("Registros de dispositivos carregados com sucesso.")
        except Exception as e:
            print(f"Erro ao carregar registros: {e}")
            registrations = {}
    else:
        registrations = {}

def save_registrations():
    try:
        with open(REGISTRATIONS_FILE, "w") as f:
            with registrations_lock:
                json.dump(registrations, f, indent=4)
    except Exception as e:
        print(f"Erro ao salvar registros: {e}")

load_registrations()

# Envio de Alerta pelo Telegram
def send_telegram_alert(image_path: str, chat_id: int, dispositivo_id: str):
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN não configurado. Impossível enviar alerta.")
        return

    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    caption = (
        f"🚨 *ALERTA DE SEGURANÇA* 🚨\n\n"
        f"Um Animal Peçonhento foi detectado pelo dispositivo:\n"
        f"🆔 `{dispositivo_id}`\n\n"
        f"⚠️ Cuidado! Verifique o local imediatamente."
    )
    
    try:
        with open(image_path, "rb") as photo_file:
            files = {"photo": photo_file}
            data = {
                "chat_id": chat_id,
                "caption": caption,
                "parse_mode": "Markdown"
            }
            res = requests.post(url_photo, data=data, files=files, timeout=20)
            if res.status_code != 200:
                print(f"Erro ao enviar alerta via Telegram: {res.text}")
    except Exception as e:
        print(f"Exceção ao enviar e-mail/mensagem Telegram: {e}")

# Thread de Polling para o Bot do Telegram
def telegram_bot_polling():
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN não encontrado. Bot desativado.")
        return

    print("Iniciando Polling do Bot do Telegram...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/"
    last_update_id = 0
    
    # Obter o último update_id pendente para não processar mensagens antigas
    try:
        res = requests.get(f"{url}getUpdates", params={"limit": 1, "timeout": 1}, timeout=5)
        if res.status_code == 200:
            updates = res.json().get("result", [])
            if updates:
                last_update_id = updates[-1]["update_id"]
    except Exception as e:
        print(f"Erro ao inicializar polling: {e}")

    while True:
        try:
            res = requests.get(f"{url}getUpdates", params={"offset": last_update_id + 1, "timeout": 20}, timeout=25)
            if res.status_code == 200:
                updates = res.json().get("result", [])
                for update in updates:
                    last_update_id = update["update_id"]
                    if "message" in update:
                        msg = update["message"]
                        chat_id = msg["chat"]["id"]
                        text = msg.get("text", "").strip()
                        
                        if text in ("/start", "➕ Registrar Novo"):
                            welcome_msg = (
                                "Olá! Bem-vindo ao *Bot VenomESP* 🦂\n\n"
                                "Para receber alertas de escorpiões do seu ESP32-CAM, "
                                "por favor me envie o *Número de Série* do seu dispositivo.\n\n"
                                "Exemplo:\n`ESP32-CAM-XX:XX:XX:XX:XX:XX`"
                            )
                            requests.post(f"{url}sendMessage", json={
                                "chat_id": chat_id, 
                                "text": welcome_msg, 
                                "parse_mode": "Markdown",
                                "reply_markup": {
                                    "keyboard": [
                                        [{"text": "📋 Meus Dispositivos"}, {"text": "➕ Registrar Novo"}],
                                        [{"text": "❓ Ajuda"}]
                                    ],
                                    "resize_keyboard": True
                                }
                            })
                        elif text == "📋 Meus Dispositivos":
                            with registrations_lock:
                                user_devices = [dev for dev, cid in registrations.items() if cid == chat_id]
                            if user_devices:
                                devices_list = "\n".join(f"• `{dev}`" for dev in user_devices)
                                msg_text = f"📋 *Seus Dispositivos Cadastrados:*\n\n{devices_list}\n\nVocê receberá alertas em tempo real para todos eles!"
                            else:
                                msg_text = "Você ainda não possui nenhum dispositivo VenomESP cadastrado.\nEnvie o número de série para cadastrar!"
                            requests.post(f"{url}sendMessage", json={
                                "chat_id": chat_id, 
                                "text": msg_text, 
                                "parse_mode": "Markdown"
                            })
                        elif text == "❓ Ajuda":
                            help_msg = (
                                "🦂 *Ajuda - Sistema VenomESP*\n\n"
                                "Este bot recebe alertas de imagens do seu ESP32-CAM sempre que um escorpião ou animal peçonhento for detectado pela nossa inteligência artificial.\n\n"
                                "• Para cadastrar um dispositivo, basta digitar o número de série completo (ex: `ESP32-CAM-XX:XX:XX:XX:XX:XX`).\n"
                                "• Você pode cadastrar múltiplos dispositivos para este mesmo chat.\n"
                                "• Quando o ESP32-CAM inicializar, ele piscará o flash uma vez e mandará uma foto de teste para você confirmar o funcionamento."
                            )
                            requests.post(f"{url}sendMessage", json={
                                "chat_id": chat_id, 
                                "text": help_msg, 
                                "parse_mode": "Markdown"
                            })
                        elif text.startswith("ESP32-CAM-"):
                            dispositivo_id = text
                            with registrations_lock:
                                registrations[dispositivo_id] = chat_id
                            save_registrations()
                            
                            confirm_msg = (
                                f"✅ *Dispositivo registrado com sucesso!*\n\n"
                                f"Dispositivo ID: `{dispositivo_id}`\n"
                                f"Chat ID: `{chat_id}`\n\n"
                                "Você receberá alertas em tempo real sempre que um perigo for detectado neste dispositivo!"
                            )
                            requests.post(f"{url}sendMessage", json={
                                "chat_id": chat_id, 
                                "text": confirm_msg, 
                                "parse_mode": "Markdown",
                                "reply_markup": {
                                    "keyboard": [
                                        [{"text": "📋 Meus Dispositivos"}, {"text": "➕ Registrar Novo"}],
                                        [{"text": "❓ Ajuda"}]
                                    ],
                                    "resize_keyboard": True
                                }
                            })
                        elif text:
                            invalid_msg = (
                                "⚠️ *Comando ou formato inválido.*\n\n"
                                "Para registrar seu dispositivo, envie o número de série no formato:\n"
                                "`ESP32-CAM-XX:XX:XX:XX:XX:XX`"
                            )
                            requests.post(f"{url}sendMessage", json={
                                "chat_id": chat_id, 
                                "text": invalid_msg, 
                                "parse_mode": "Markdown"
                            })
        except Exception as e:
            print(f"Erro no loop do Telegram Bot: {e}")
        time.sleep(2)

# Iniciar thread do Bot do Telegram
polling_thread = threading.Thread(target=telegram_bot_polling, daemon=True)
polling_thread.start()

# Thread de Keep-Alive para evitar suspensão no Render
def keep_alive_ping():
    # Aguarda a API subir completamente
    time.sleep(15)
    print("Iniciando monitoramento de Keep-Alive para Render.com...")
    while True:
        try:
            if API_URL and API_URL.startswith("http"):
                # Faz um GET simples na raiz da própria API
                res = requests.get(API_URL, timeout=10)
                print(f"Keep-Alive ping enviado para {API_URL}. Status: {res.status_code}")
        except Exception as e:
            print(f"Erro no ping de Keep-Alive: {e}")
        # Dorme por 10 minutos (600 segundos)
        time.sleep(600)

keep_alive_thread = threading.Thread(target=keep_alive_ping, daemon=True)
keep_alive_thread.start()

class DeteccaoResponse(BaseModel):
    animal_detectado: bool
    acionar_alarme: bool
    tempo_segundos: int | None = None
    erro: str | None = None

@app.get("/")
async def root():
    return {"status": "API Online", "documentacao": "/docs", "telegram_bot": "Ativo"}

@app.get("/config", tags=["Configuração"])
async def get_config():
    bot_username = "BotDesconhecido"
    if TELEGRAM_BOT_TOKEN:
        try:
            res = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe", timeout=5)
            if res.status_code == 200:
                bot_username = res.json().get("result", {}).get("username", "BotDesconhecido")
        except Exception:
            pass
    return {"bot_username": bot_username}

@app.get("/status-dispositivo/{dispositivo_id}", tags=["Status"])
async def status_dispositivo(dispositivo_id: str):
    with registrations_lock:
        is_registered = dispositivo_id in registrations
    return {
        "dispositivo_id": dispositivo_id,
        "registrado": is_registered,
        "api_online": True
    }

@app.post("/detectar", response_model=DeteccaoResponse, tags=["Detecção"])
def detectar_animal(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    dispositivo_id: str = Form(...),
    is_test: bool = Form(False)
):
    if not dispositivo_id:
        raise HTTPException(status_code=400, detail="O dispositivo_id é obrigatório.")

    try:
        contents = file.file.read()
        
        # Se for uma foto de teste
        if is_test:
            imagem_alerta_nome = f"teste_{dispositivo_id.replace(':', '_')}_{int(time.time())}.jpg"
            imagem_alerta_path = os.path.join("static", imagem_alerta_nome)
            with open(imagem_alerta_path, "wb") as f:
                f.write(contents)
            
            # Verificar se há chat cadastrado
            with registrations_lock:
                chat_id = registrations.get(dispositivo_id)
            
            if chat_id:
                def send_test_alert(image_path: str, chat_id: int, dispositivo_id: str):
                    if not TELEGRAM_BOT_TOKEN:
                        return
                    url_photo = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                    caption = (
                        f"📸 *CONEXÃO ESTABELECIDA* 📸\n\n"
                        f"O dispositivo `{dispositivo_id}` foi inicializado com sucesso e enviou esta primeira imagem de teste!\n\n"
                        f"🟢 Status: *Online e Monitorando*"
                    )
                    try:
                        with open(image_path, "rb") as photo_file:
                            files = {"photo": photo_file}
                            data = {
                                "chat_id": chat_id,
                                "caption": caption,
                                "parse_mode": "Markdown"
                            }
                            requests.post(url_photo, data=data, files=files, timeout=20)
                    except Exception as e:
                        print(f"Erro ao enviar foto de teste: {e}")
                
                background_tasks.add_task(send_test_alert, imagem_alerta_path, chat_id, dispositivo_id)
                return DeteccaoResponse(
                    animal_detectado=False,
                    acionar_alarme=False,
                    tempo_segundos=0
                )
            else:
                return DeteccaoResponse(
                    animal_detectado=False,
                    acionar_alarme=False,
                    erro=f"Dispositivo {dispositivo_id} não está registrado no bot do Telegram."
                )

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Falha ao decodificar a imagem.")
        
        if model is None:
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Modelo YOLOv8 não carregado.")
            
        results = model.predict(source=img, conf=0.4, imgsz=320, save=False)
        result = results[0]
        deteccoes = result.boxes
        
        if len(deteccoes) > 0:
            # Plota os bounding boxes
            img_com_boxes = result.plot()
            imagem_alerta_nome = f"alerta_{int(time.time())}.jpg"
            imagem_alerta_path = os.path.join("static", imagem_alerta_nome)
            cv2.imwrite(imagem_alerta_path, img_com_boxes)
            
            # Verificar se há chat cadastrado
            with registrations_lock:
                chat_id = registrations.get(dispositivo_id)
            
            if chat_id:
                # Dispara notificação Telegram em segundo plano
                background_tasks.add_task(send_telegram_alert, imagem_alerta_path, chat_id, dispositivo_id)
            else:
                print(f"Alerta gerado, mas nenhum Chat ID Telegram registrado para o dispositivo {dispositivo_id}")
            
            return DeteccaoResponse(
                animal_detectado=True,
                acionar_alarme=True,
                tempo_segundos=15
            )
        else:
            return DeteccaoResponse(
                animal_detectado=False,
                acionar_alarme=False
            )
            
    except Exception as e:
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro=str(e))

@app.post("/detectar-url", response_model=DeteccaoResponse, tags=["Detecção via URL"])
async def detectar_animal_url(
    background_tasks: BackgroundTasks, 
    image_url: str = Form(...),
    dispositivo_id: str = Form(...)
):
    if not dispositivo_id:
        raise HTTPException(status_code=400, detail="O dispositivo_id é obrigatório.")
    if not image_url:
        raise HTTPException(status_code=400, detail="A image_url é obrigatória.")

    try:
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        contents = response.content

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Falha ao decodificar a imagem da URL.")
        
        if model is None:
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Modelo YOLOv8 não carregado.")
            
        results = model.predict(source=img, conf=0.4, imgsz=320, save=False)
        result = results[0]
        deteccoes = result.boxes
        
        if len(deteccoes) > 0:
            img_com_boxes = result.plot()
            imagem_alerta_nome = f"alerta_url_{int(time.time())}.jpg"
            imagem_alerta_path = os.path.join("static", imagem_alerta_nome)
            cv2.imwrite(imagem_alerta_path, img_com_boxes)
            
            with registrations_lock:
                chat_id = registrations.get(dispositivo_id)
            
            if chat_id:
                background_tasks.add_task(send_telegram_alert, imagem_alerta_path, chat_id, dispositivo_id)
            
            return DeteccaoResponse(
                animal_detectado=True,
                acionar_alarme=True,
                tempo_segundos=15
            )
        else:
            return DeteccaoResponse(
                animal_detectado=False,
                acionar_alarme=False
            )
            
    except Exception as e:
        return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
