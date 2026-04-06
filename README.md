# API YOLOv8 - Detecção de Escorpiões (ESP32-CAM)

Esta é a API final responsável por receber as imagens do ESP32-CAM, analisar com o modelo YOLOv8 já treinado (`best.pt`) e enviar um alerta por E-mail caso um escorpião seja detectado.

## Resultados e Performance do Modelo Treinado

O nosso modelo foi treinado e avaliado com foco em **alta velocidade** e **precisão**, ideal para ser utilizado junto com dispositivos IoT de recursos limitados (como o ESP32-CAM). Abaixo estão os resultados detalhados obtidos durante a validação da nossa inteligência artificial:

### Arquitetura e Processamento
- **73 camadas**: A rede neural possui 73 camadas em sua arquitetura.
- **3.005.843 parâmetros**: O modelo possui aproximadamente 3 milhões de pesos. Isso indica um modelo bem leve (característico da arquitetura YOLOv8 Nano - `yolov8n`), projetado para processamento rápido.
- **8.1 GFLOPs**: Custo computacional baixo, estimado em 8.1 bilhões de operações por segundo.

### Métricas de Precisão (Acurácia)
Durante a fase de validação (usando 475 imagens com 486 instâncias/escorpiões), obtivemos resultados impressionantes:
- **Precisão (P) - 93,4%**: De todas as vezes que o modelo alertou "Há um escorpião", ele estava correto em 93,4% das vezes. Isso significa uma taxa baixíssima de falsos positivos.
- **Revocação (R) - 90,5%**: De todos os escorpiões que *realmente* existiam nas imagens, o modelo conseguiu encontrar 90,5% deles. Uma taxa de falsos negativos excelente.
- **mAP50 - 92,2%**: Atingiu uma precisão média de 92,2% ao considerar acertos com uma sobreposição (IoU) de pelo menos 50% com o objeto real.
- **mAP50-95 - 70,3%**: Uma métrica extremamente rigorosa, mostrando que as caixas delimitadoras (bounding boxes) desenhadas pela nossa IA não apenas detectam, mas ficam **muito bem alinhadas** com as bordas reais do escorpião.

### Métricas de Velocidade (Por Imagem)
- **Pré-processamento**: 0.2ms (Redimensionamento e normalização).
- **Inferência**: 2.2ms (O processamento neural profundo. É extremamente rápido!).
- **Pós-processamento**: 1.0ms (Remoção de caixas sobrepostas via NMS).

---

## Como rodar a API localmente

1. Tenha certeza de que o arquivo `best.pt` (os pesos do modelo) está na mesma pasta deste `README.md`.
2. Crie um arquivo `.env` na raiz do projeto com as credenciais do seu e-mail (usadas para disparar os alertas):
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
5. Acesse a documentação e interface interativa Swagger em: **`http://localhost:8000/docs`**

## Parâmetros da Requisição (Como usar)

A API espera uma requisição POST na rota `/detectar` com o formato `multipart/form-data` contendo:
- `file`: O arquivo de imagem (foto capturada pelo ESP32-CAM) a ser analisado.
- `email_destinatario`: O endereço de e-mail da pessoa que receberá o alerta caso um escorpião seja detectado na imagem enviada.

## Deploy na Nuvem (Ex: Render)

O repositório já está configurado com o `.gitignore` para subir apenas os arquivos estritamente necessários para a API rodar no Render (evitando estourar limites com arquivos pesados de datasets).

Os arquivos que irão para o seu GitHub são:
- `main.py`
- `best.pt` (Modelo)
- `requirements.txt`
- `Procfile`
- `.gitignore`
- `README.md`

No Render, crie um "Web Service", conecte seu repositório GitHub e use as configurações:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**⚠️ IMPORTANTE**: Você **deve** configurar as Variáveis de Ambiente (`EMAIL_REMETENTE` e `EMAIL_SENHA`) diretamente no painel do Render (na aba *Environment*). O arquivo `.env` nunca é enviado para o GitHub por razões óbvias de segurança.
