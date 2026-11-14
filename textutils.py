# textutils.py
"""Чистая пост-обработка ответов модели.

Вынесено из main.py, чтобы функции можно было тестировать без загрузки
sentence-transformers/torch и без обращения к Ollama.
"""
import re

# Сколько предложений оставляем в ответе клиенту для каждого режима.
MAX_SENTENCES_BY_MODE = {
    "pizza_staff": 2,
}
DEFAULT_MAX_SENTENCES = 4


def truncate_client_answer(text: str, mode: str = "") -> str:
    """Вычищает маркдаун и обрезает ответ до N предложений.

    Бот иногда пишет поэмы, списки и официальные подписи — в чат поддержки
    это не годится.
    """
    if not text:
        return ""

    text = text.replace("**", "").replace("*", "")
    text = re.sub(r'(?i)с уважением.*', '', text)
    text = re.sub(r'(?m)^\d+[\.\)]\s*', '', text)
    text = re.sub(r'(?m)^[-•]\s*', '', text)
    text = " ".join(line.strip() for line in text.split('\n') if line.strip())

    max_sentences = MAX_SENTENCES_BY_MODE.get(mode, DEFAULT_MAX_SENTENCES)
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > max_sentences:
        text = " ".join(sentences[:max_sentences])
        if not text.endswith(('.', '!', '?')):
            text += "..."
    return text.strip()
