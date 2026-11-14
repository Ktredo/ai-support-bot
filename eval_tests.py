# eval_tests.py
"""End-to-end evals: прогоняют НАСТОЯЩИЙ пайплайн (process_query) и оценивают ответы.

Раньше здесь была копия логики из main.py, которая разъехалась с ним и перестала
запускаться. Теперь evals вызывают то же, что получает пользователь, — иначе
они проверяют не продукт, а сами себя.
"""
import os
import logging
import requests
from dotenv import load_dotenv
from main import SupportSystem

load_dotenv()
LOCAL_URL = os.getenv("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen2.5:7b")
EVAL_MODE = os.getenv("EVAL_MODE", "bank_support")

# --- ИЗОЛИРОВАННЫЙ ЛОГГЕР ДЛЯ ТЕСТОВ ---
eval_logger = logging.getLogger("EvalLogger")
eval_logger.setLevel(logging.INFO)
eval_logger.propagate = False  # Отключаем передачу в корневой логгер (чтобы не было дублей)

fh = logging.FileHandler("eval_results.log", mode='w', encoding='utf-8')
ch = logging.StreamHandler()

formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

eval_logger.addHandler(fh)
eval_logger.addHandler(ch)
# ----------------------------------------

TEST_CASES = [
    "Почему в Ozon Банке арест, а в других нет?",
    "Можно ли закрыть счет, если там есть деньги и арест?",
    "Что такое ГИСП?",
    "Сколько ждать снятия ареста от ФНС?",
]


def judge_answer(question, bot_answer, context):
    prompt = f"""Ты Senior QA в банке. Оцени ответ бота от 1 до 10.
Вопрос клиента: {question}
Ответ бота: {bot_answer}
Контекст базы знаний: {context}

Критерии:
1. Ответ вежливый?
2. Если ответ содержит фразу "Я уточню информацию", и в контексте ДЕЙСТВИТЕЛЬНО нет ответа — ставь 9/10 (это правильный фоллбэк).
3. Если ответ опирается на факты из контекста — ставь 8-10/10.
4. Если бот выдумал факты, которых нет в контексте — ставь 1/10.

Верни ТОЛЬКО число от 1 до 10."""

    payload = {"model": JUDGE_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    try:
        r = requests.post(LOCAL_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "0").strip()
    except Exception as e:
        eval_logger.error(f"Ошибка Судьи: {e}")
        return "0"


if __name__ == "__main__":
    system = SupportSystem()
    eval_logger.info(f"=== ЗАПУСК АВТОМАТИЧЕСКОГО ТЕСТИРОВАНИЯ (EVALS) | режим: {EVAL_MODE} ===\n")

    total_score = 0
    for i, q in enumerate(TEST_CASES):
        eval_logger.info(f"\n[{i+1}/{len(TEST_CASES)}] Вопрос: {q}")

        # У каждого кейса свой user_id: иначе история диалога из TTLCache
        # подмешивается в следующие вопросы и завышает оценки.
        query, final_answer, _note = system.process_query(f"eval_{i}", EVAL_MODE, q)

        # Контекст нужен только судье. Берем его тем же поиском, что и пайплайн.
        context = system.rag.get_context(query)

        eval_logger.info(f"[Поисковый запрос]: {query}")
        eval_logger.info(f"[Финальный ответ]: {final_answer}")

        score_str = judge_answer(q, final_answer, context)
        try:
            score = float(score_str)
        except ValueError:
            eval_logger.error(f"Судья вернул не число: {score_str!r}")
            score = 0.0

        total_score += score
        eval_logger.info(f"[Оценка QA]: {score}/10")
        eval_logger.info("-" * 50)

    eval_logger.info(f"\n=== СРЕДНИЙ БАЛЛ: {total_score / len(TEST_CASES)}/10 ===")
    eval_logger.info("Результаты сохранены в файл eval_results.log")

    fh.flush()
