# api.py
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import prompts

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("API")

# Глобальная ссылка на систему. Заполняется в lifespan (или подменяется в тестах).
system: Any = None


def load_system() -> Any:
    """Создает SupportSystem.

    Импорт намеренно ленивый: main тянет sentence-transformers/torch,
    и без него модуль api импортируется мгновенно (важно для тестов).
    """
    from main import SupportSystem

    return SupportSystem()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Загружаем модели и базу знаний ОДИН РАЗ при старте сервера."""
    global system
    logger.info("Инициализация SupportSystem...")
    system = load_system()
    logger.info("Система готова к приему запросов!")
    try:
        yield
    finally:
        system = None


app = FastAPI(
    title="AI Support Bot API",
    description="API для умного ассистента поддержки (RAG + Self-Correction)",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Модели данных (Pydantic) ---
class ChatRequest(BaseModel):
    client_question: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(default="anonymous", max_length=128)
    mode: str = "bank_support"
    account_info: str = Field(default="Нет данных", max_length=2000)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        # Единственный источник правды — prompts.MODES, чтобы список режимов
        # не расходился между CLI и API.
        if value not in prompts.MODES:
            raise ValueError(
                f"Неизвестный режим: {value}. Доступно: {', '.join(prompts.MODES)}"
            )
        return value


class ChatResponse(BaseModel):
    search_query: str
    client_answer: str
    operator_note: str


# --- Эндпоинты ---
@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Основной эндпоинт для общения с ботом.

    Обычный def, а не async def: process_query блокирующий (requests + torch),
    поэтому FastAPI выполняет его в пуле потоков и не блокирует event loop.
    """
    if system is None:
        raise HTTPException(status_code=503, detail="Система еще загружается...")

    try:
        query, answer, note = system.process_query(
            request.user_id,
            request.mode,
            request.client_question,
            request.account_info,
        )
    except Exception:
        logger.exception("Ошибка при обработке запроса")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

    return ChatResponse(
        search_query=query,
        client_answer=answer,
        operator_note=note,
    )


@app.get("/modes")
def modes():
    """Список доступных режимов (для клиента/Telegram-бота)."""
    return {key: data["name"] for key, data in prompts.MODES.items()}


@app.get("/health")
def health_check():
    """Проверка работоспособности сервера (для Docker/деплоя)."""
    return {"status": "ok", "system_loaded": system is not None}
