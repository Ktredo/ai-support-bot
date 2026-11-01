# eval_tests.py
import os
import logging
import requests
from dotenv import load_dotenv
from main import SupportSystem
import prompts

load_dotenv()
LOCAL_URL = os.getenv("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen2.5:7b")

# --- ИЗОЛИРОВАННЫЙ ЛОГГЕР ДЛЯ ТЕСТОВ ---
eval_logger = logging.getLogger("EvalLogger")
eval_logger.setLevel(logging.INFO)
eval_logger.propagate = False # Отключаем передачу в корневой логгер (чтобы не было дублей)

# Создаем обработчики (один пишет в файл, второй в консоль)
fh = logging.FileHandler("eval_results.log", mode='w', encoding='utf-8')
ch = logging.StreamHandler()

# Формат
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
    "Сколько ждать снятия ареста от ФНС?"
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
    
    # Отправляем ЛОКАЛЬНОЙ модели
    payload = {"model": JUDGE_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    try:
        r = requests.post(LOCAL_URL, json=payload, timeout=60)
        return r.json().get("response", "0").strip()
    except Exception as e:
        logging.error(f"Ошибка Судьи: {e}")
        return "0"

if __name__ == "__main__":
    system = SupportSystem()
    eval_logger.info("=== ЗАПУСК АВТОМАТИЧЕСКОГО ТЕСТИРОВАНИЯ (EVALS) ===\n")
    
    total_score = 0
    for i, q in enumerate(TEST_CASES):
        eval_logger.info(f"\n[{i+1}/{len(TEST_CASES)}] Вопрос: {q}")
        
        # 1. Поиск
        context = system.rag.get_context(q)
        eval_logger.info(f"--- НАЙДЕННЫЙ КОНТЕКСТ ---\n{context}\n---------------------------")
        
        # 2. Генерация
        prompt_text = prompts.CLIENT_ANSWER_PROMPT.format(
            tov_text=system.tov_text, context_text=context, emotion="нейтральный", client_question=q
        )
        raw_answer = system._ask_llm(prompt_text, temperature=0.5)
        raw_answer = system._clean_text(raw_answer)
        eval_logger.info(f"[Сырой ответ бота]: {raw_answer}")
        
        # 3. Аудит
        is_safe, feedback = system.auditor.audit(raw_answer, context)
        
        if not is_safe:
            eval_logger.info(f"[Аудитор]: ЗАБРАКОВАНО. Причина: {feedback}")
            final_answer = "Я уточню информацию по вашему вопросу и вернусь с ответом."
        else:
            eval_logger.info(f"[Аудитор]: ОДОБРЕНО.")
            final_answer = raw_answer

        # 4. Оценка
        score_str = judge_answer(q, final_answer, context)
        try:
            score = float(score_str)
        except:
            score = 0.0
            
        total_score += score
        eval_logger.info(f"[Финальный ответ]: {final_answer}")
        eval_logger.info(f"[Оценка QA]: {score}/10")
        eval_logger.info("-" * 50)

    eval_logger.info(f"\n=== СРЕДНИЙ БАЛЛ: {total_score / len(TEST_CASES)}/10 ===")
    eval_logger.info("Результаты сохранены в файл eval_results.log")
    
    # Принудительно сбрасываем буфер в файл на диск
    fh.flush()