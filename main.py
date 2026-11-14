# main.py
import os
import re
import time
import logging
import requests
from dotenv import load_dotenv
from qdrant_rag import QdrantRAGEngine
from auditor import FactAuditor
import prompts
from textutils import truncate_client_answer
from cachetools import TTLCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Orchestrator")

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

class SupportSystem:
    def __init__(self):
        kb_path = os.path.abspath("knowledge_base.txt")
        self.rag = QdrantRAGEngine(kb_path)
        self.sessions = TTLCache(maxsize=1000, ttl=300)

        local_url = os.getenv("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
        auditor_model = os.getenv("AUDITOR_MODEL", "qwen2.5:7b")
        self.auditor = FactAuditor(local_url, auditor_model)

    def get_memory(self, user_id: str):
        if user_id not in self.sessions:
            self.sessions[user_id] = []
        return self.sessions[user_id]

    def _add_to_memory(self, user_id: str, role: str, text: str):
        mem = self.get_memory(user_id)
        mem.append(f"{role}: {text}")
        if len(mem) > 6:
            mem = mem[-6:]
        self.sessions[user_id] = mem

    def _ask_llm(self, user_prompt: str, system_prompt: str = "Ты полезный ассистент.", temperature: float = 0.6) -> str:
        payload = {
            "model": MODEL_NAME, "system": system_prompt, "prompt": user_prompt, "stream": False,
            "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1000, "stop": ["<|im_end|>", "</s>", "[INST]"]},
        }
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json().get("response", "")
        except Exception as e:
            logger.error(f"Ошибка API: {e}")
            return ""

    def _truncate_client_answer(self, text: str, mode: str) -> str:
        """Обертка для обратной совместимости: логика живет в textutils."""
        return truncate_client_answer(text, mode)

    def process_query(self, user_id: str, mode: str, client_question: str, account_info: str = "Нет данных") -> tuple[str, str, str]:
        """Triage -> RAG -> генерация -> самопроверка.

        Возвращает (поисковый запрос, ответ клиенту, заметку оператору).
        """
        logger.info(f"Анализ запроса от {user_id} (Режим: {mode})...")

        history_text = "\n".join(self.get_memory(user_id))
        if not history_text: history_text = "История пуста."

        mode_data = prompts.MODES[mode]

        # 1. Triage
        triage_prompt = f"""Ты работаешь в пиццерии.
{mode_data['triage_instruction']}
История: {history_text}
CRM: {account_info}
Вопрос: {client_question}

Ответь СТРОГО в формате:
ЗАПРОС: [суть вопроса]
"""
        triage_result = self._ask_llm(triage_prompt, temperature=0.1).strip()

        search_query_match = re.search(r'ЗАПРОС:\s*(.*)', triage_result)
        search_query = search_query_match.group(1).strip() if search_query_match else client_question

        logger.info(f"Поисковый запрос: {search_query}")
        context_text = self.rag.get_context(search_query)

        # 2. Генерация
        logger.info("Генерация ответа...")
        client_prompt = f"""{mode_data['client_instruction']}
ToV: {mode_data['tov']}
Контекст: {context_text}
CRM: {account_info}
Вопрос: {client_question}

Твой ответ:"""
        client_answer = self._ask_llm(client_prompt, temperature=0.5)
        client_answer = self._truncate_client_answer(client_answer, mode)

        # 3. Self-Correction
        max_retries = 2
        for attempt in range(max_retries):
            is_safe, feedback = self.auditor.audit(client_answer, context_text)
            if is_safe:
                break

            logger.warning(f"Аудитор забраковал (Попытка {attempt+1}): {feedback}")
            if attempt == max_retries - 1:
                client_answer = "Я уточню информацию и вернусь с ответом."
                break

            fix_prompt = f"""Ты написала ответ, но аудитор забраковал его.
Ошибка: {feedback}
Старый ответ: {client_answer}
База: {context_text}

Перепиши ответ, используя ТОЛЬКО факты из базы (2-3 предложения)."""
            client_answer = self._ask_llm(fix_prompt, temperature=0.3)
            client_answer = self._truncate_client_answer(client_answer, mode)

        self._add_to_memory(user_id, "Клиент", client_question)
        self._add_to_memory(user_id, "Ты", client_answer)

        # 4. Заметка оператору
        op_prompt = f"""{mode_data['operator_note']}
        """.format(
            client_question=client_question,
            client_answer=client_answer,
            context_text=context_text,
            account_info=account_info
        )

        op_note = self._ask_llm(op_prompt, temperature=0.85).strip()
        op_note = op_note.replace("**", "").replace("*", "")

        return search_query, client_answer, op_note


if __name__ == "__main__":
    system = SupportSystem()
    current_mode = "bank_support" # По умолчанию банк

    logger.info(f"=== СИСТЕМА ГОТОВА (Режим: {prompts.MODES[current_mode]['name']}) ===")
    logger.info("Доступные режимы: " + ", ".join(f"/mode {key}" for key in prompts.MODES))

    while True:
        try:
            user_id = input("\nВведите ID клиента (Enter - 123): ") or "123"
            if user_id.lower() in ["выход", "exit"]: break

            q = input(f"Клиент [{prompts.MODES[current_mode]['name']}]: ")

            # Проверяем смену режима
            if q.startswith("/mode "):
                new_mode = q.split(" ")[1].strip()
                if new_mode in prompts.MODES:
                    current_mode = new_mode
                    print(f"\n[Система] Режим переключен на: {prompts.MODES[current_mode]['name']}")
                else:
                    print(f"\n[Система] Неизвестный режим. Доступно: {', '.join(prompts.MODES.keys())}")
                continue

            if q.lower() in ["выход", "exit"]: break

            acc_info = input("Доп. инфо (Enter, если пусто): ")
            if not acc_info: acc_info = "Нет данных."

            query_or_flag, ans, note = system.process_query(user_id, current_mode, q, acc_info)

            print(f"\n[Поисковый запрос]: {query_or_flag}")
            print(f"\n[Система] Бот ищет ответ...")
            time.sleep(1)

            print(f"\n[Ответ]:")
            for char in ans:
                print(char, end="", flush=True)
                time.sleep(0.01)
            print()

            print(f"\n[Заметка для тебя]:\n{note}")
            print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
