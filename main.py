import os
import cv2
import numpy as np
import smtplib
from email.message import EmailMessage
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="API Detecção de Escorpiões - ESP32Cam",
    description="API que recebe imagens de um ESP32, analisa através do YOLOv8 treinado localmente se há escorpiões e envia alertas por E-mail.",
    version="1.0.0",
    docs_url="/docs",  
    redoc_url="/redoc" 
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

YOLO_MODEL_PATH = "best.pt"

EMAIL_REMETENTE = os.getenv("EMAIL_REMETENTE")
EMAIL_SENHA = os.getenv("EMAIL_SENHA")

if not EMAIL_REMETENTE or not EMAIL_SENHA:
    print("Aviso: Variáveis de ambiente EMAIL_REMETENTE ou EMAIL_SENHA não configuradas.")

try:
    if os.path.exists(YOLO_MODEL_PATH):
        model = YOLO(YOLO_MODEL_PATH)
        print(f"Modelo YOLO carregado com sucesso a partir de {YOLO_MODEL_PATH}!")
    else:
        print(f"Aviso: Arquivo de modelo não encontrado em {YOLO_MODEL_PATH}.")
        model = None
except Exception as e:
    print(f"Aviso: Erro ao inicializar o YOLOv8: {e}")
    model = None

def send_email_alert(image_path: str, destinatario: str):
    if not EMAIL_REMETENTE or not EMAIL_SENHA:
        print("Erro: Remetente ou senha do e-mail não configurados. O alerta não foi enviado.")
        return

    try:
        msg = EmailMessage()
        msg['Subject'] = '🚨 ALERTA: Escorpião Detectado!'
        msg['From'] = EMAIL_REMETENTE
        msg['To'] = destinatario
        msg.set_content('Um escorpião foi detectado pela câmera do ESP32. Por favor, analise a imagem em anexo.')

        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_name = os.path.basename(image_path)
            image_type = 'jpeg'
            
        msg.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_REMETENTE, EMAIL_SENHA)
            smtp.send_message(msg)
            
        print(f"E-mail de alerta enviado com sucesso para {destinatario}!")
    except Exception as e:
        print(f"Erro ao enviar e-mail: {e}")

class DeteccaoResponse(BaseModel):
    animal_detectado: bool
    acionar_alarme: bool
    tempo_segundos: int | None = None
    erro: str | None = None

@app.post("/detectar", response_model=DeteccaoResponse, summary="Detecta escorpiões em uma imagem enviada pelo ESP32", tags=["Detecção"])
async def detectar_animal(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    email_destinatario: str = Form(...)
):
    if not email_destinatario:
        raise HTTPException(status_code=400, detail="O email_destinatario é obrigatório.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if model is None:
            return DeteccaoResponse(animal_detectado=False, acionar_alarme=False, erro="Modelo YOLOv8 não carregado.")
            
        results = model.predict(source=img, conf=0.4, save=False)
        result = results[0]
        deteccoes = result.boxes
        
        if len(deteccoes) > 0:
            img_com_boxes = result.plot()
                
            imagem_alerta_nome = "alerta_atual.jpg"
            imagem_alerta_path = os.path.join("static", imagem_alerta_nome)
            cv2.imwrite(imagem_alerta_path, img_com_boxes)
            
            background_tasks.add_task(send_email_alert, imagem_alerta_path, email_destinatario)
            
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
