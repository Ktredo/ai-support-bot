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

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("Orchestrator")

load_dotenv()
OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
TOV_FILE = "knowledge_base/tov_rules.md"

class SupportSystem:
    def __init__(self):
        kb_path = os.path.abspath("knowledge_base.txt")
        self.rag = QdrantRAGEngine(kb_path)
        
        local_url = os.getenv("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
        auditor_model = os.getenv("AUDITOR_MODEL", "qwen2.5:7b")
        self.auditor = FactAuditor(local_url, auditor_model)
        
        self.tov_text = "Правила ToV не найдены."
        if os.path.exists(TOV_FILE):
            with open(TOV_FILE, "r", encoding="utf-8") as f:
                self.tov_text = f.read()

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

    def _truncate_client_answer(self, text: str) -> str:
        """Безжалостно вырезает списки, маркдаун и подписи, оставляя один абзац"""
        if not text: return ""
        
        # Убираем звездочки
        text = text.replace("**", "").replace("*", "")
        
        # Убираем подписи типа "С уважением"
        text = re.sub(r'(?i)с уважением.*', '', text)
        
        # Убираем списки (1. , 2) , - )
        text = re.sub(r'(?m)^\d+[\.\)]\s*', '', text)
        text = re.sub(r'(?m)^[-•]\s*', '', text)
        
        # Склеиваем все строки в один абзац
        text = " ".join([line.strip() for line in text.split('\n') if line.strip()])
        
        # Обрезаем, если больше 4 предложений
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) > 4:
            text = " ".join(sentences[:4])
            if not text.endswith(('.', '!', '?')):
                text += "..."
                
        return text.strip()

    def process_query(self, client_question: str):
        logger.info("Анализ запроса...")
        triage_result = self._ask_llm(prompts.TRIAGE_PROMPT + f"\n\nКлиент: {client_question}", temperature=0.1).strip()
        
        if "УТОЧНИТЬ:" in triage_result:
            clarify_text = triage_result.split("УТОЧНИТЬ:")[1].strip()
            return "УТОЧНЕНИЕ", clarify_text, "Клиент написал непонятно. Я переспросила."

        search_query = re.search(r'ЗАПРОС:\s*(.*)', triage_result)
        search_query = search_query.group(1).strip() if search_query else client_question
        logger.info(f"Поисковый запрос: {search_query}")

        context_text = self.rag.get_context(search_query)

        logger.info("Генерация ответа...")
        client_prompt = prompts.CLIENT_ANSWER_PROMPT.format(
            tov_text=self.tov_text, context_text=context_text, emotion="нейтральный", client_question=client_question
        )
        client_answer = self._ask_llm(client_prompt, temperature=0.5)
        client_answer = self._truncate_client_answer(client_answer) # ЖЕСТКАЯ ОБРЕЗКА

        # Self-Correction
        max_retries = 2
        for attempt in range(max_retries):
            is_safe, feedback = self.auditor.audit(client_answer, context_text)
            if is_safe:
                logger.info("Аудитор одобрил ответ.")
                break
                
            logger.warning(f"Аудитор забраковал (Попытка {attempt+1}): {feedback}")
            if attempt == max_retries - 1:
                client_answer = "Я уточню информацию по вашему вопросу и вернусь с ответом."
                break
                
            logger.info("Отправляем фидбек боту для переписывания...")
            fix_prompt = prompts.FIX_PROMPT.format(feedback=feedback, client_answer=client_answer, context_text=context_text)
            client_answer = self._ask_llm(fix_prompt, temperature=0.3)
            client_answer = self._truncate_client_answer(client_answer) # ОБРЕЗКА ПОСЛЕ ПЕРЕПИСЫВАНИЯ

        logger.info("Написание заметки для оператора...")
        op_prompt = prompts.OPERATOR_NOTE_PROMPT.format(
            client_question=client_question, client_answer=client_answer, context_text=context_text
        )
        op_note = self._ask_llm(op_prompt, temperature=0.85).strip()
        op_note = op_note.replace("**", "").replace("*", "") # Чистим заметку от звездочек

        return search_query, client_answer, op_note

if __name__ == "__main__":
    system = SupportSystem()
    logger.info("=== СИСТЕМА ГОТОВА ===")
    
    while True:
        try:
            q = input("\nКлиент: ")
            if q.lower() in ["выход", "exit"]: 
                break
            
            query_or_flag, ans, note = system.process_query(q)
            
            if query_or_flag == "УТОЧНЕНИЕ":
                print(f"\n[Уточняющий вопрос]:\n{ans}")
                print(f"\n[Заметка для тебя]:\n{note}")
                print("-" * 50)
                continue
            
            print(f"\n[Поисковый запрос]: {query_or_flag}")
            print(f"\n[Система] Луна ушла проверять базу...")
            time.sleep(3)
                
            print(f"\n[Ответ клиенту]:")
            for char in ans:
                print(char, end="", flush=True)
                time.sleep(0.015)
            print() 
            
            print(f"\n[Заметка для тебя]:\n{note}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Критическая ошибка в цикле: {e}")