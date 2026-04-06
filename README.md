# API YOLOv8 - Detecção de Escorpiões (ESP32-CAM)

Esta é a API final responsável por receber as imagens do ESP32-CAM, analisar com o modelo YOLOv8 já treinado (`best.pt`) e enviar um alerta por E-mail caso um escorpião seja detectado.

## Como rodar localmente

1. Tenha certeza de que o arquivo `best.pt` está na mesma pasta deste `README.md`.
2. Crie um arquivo `.env` na raiz do projeto com as credenciais do seu e-mail:
   ```env
   EMAIL_REMETENTE=seu.email@gmail.com
   EMAIL_SENHA=sua_senha_de_app
   ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Rode a API:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
5. Acesse a documentação Swagger em: `http://localhost:8000/docs`

## Parâmetros da Requisição

A API espera uma requisição POST na rota `/detectar` com o formato `multipart/form-data` contendo:
- `file`: O arquivo de imagem a ser analisado.
- `email_destinatario`: O endereço de e-mail que receberá o alerta caso um escorpião seja detectado.

## Deploy no Render (Nuvem)

O repositório já está configurado com o `.gitignore` para subir apenas os arquivos necessários para a API rodar no Render (e não os arquivos pesados de treinamento).

Os arquivos que irão para o GitHub são:
- `main.py`
- `best.pt`
- `requirements.txt`
- `Procfile`
- `.gitignore`
- `README.md`

No Render, crie um "Web Service", conecte seu GitHub e use as configurações automáticas:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**IMPORTANTE NO RENDER**: Você deve configurar as **Environment Variables** (Variáveis de Ambiente) `EMAIL_REMETENTE` e `EMAIL_SENHA` no painel do Render (na aba Environment), pois o arquivo `.env` não é enviado para o GitHub por questões de segurança.
