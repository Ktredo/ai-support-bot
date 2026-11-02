# api.py
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import SupportSystem

# Настройка логов для Uvicorn
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("API")

app = FastAPI(
    title="AI Support Bot API",
    description="API для умного ассистента поддержки (RAG + Self-Correction)",
    version="1.0.0"
)

# Глобальная переменная для системы
system = None

@app.on_event("startup")
def startup_event():
    """Загружаем модели и базу знаний ОДИН РАЗ при старте сервера"""
    global system
    logger.info("Инициализация SupportSystem...")
    system = SupportSystem()
    logger.info("Система готова к приему запросов!")

# --- Модели данных (Pydantic) ---
class ChatRequest(BaseModel):
    client_question: str

class ChatResponse(BaseModel):
    search_query: str
    client_answer: str
    operator_note: str

# --- Эндпоинты ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Основной эндпоинт для общения с ботом"""
    if not system:
        raise HTTPException(status_code=503, detail="Система еще загружается...")
    
    try:
        query, answer, note = system.process_query(request.client_question)
        return ChatResponse(
            search_query=query,
            client_answer=answer,
            operator_note=note
        )
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера (для Docker/деплоя)"""
    return {"status": "ok", "system_loaded": system is not None}