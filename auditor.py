# auditor.py
import logging
import requests
from prompts import AUDITOR_PROMPT

logger = logging.getLogger("Auditor")

class FactAuditor:
    def __init__(self, ollama_url: str, model_name: str):
        self.ollama_url = ollama_url
        self.model_name = model_name

    # ДОБАВЛЕН АРГУМЕНТ account_info
    def audit(self, client_answer: str, context_text: str, account_info: str = "Нет данных") -> tuple[bool, str]:
        prompt = AUDITOR_PROMPT.format(
            context_text=context_text, 
            client_answer=client_answer,
            account_info=account_info
        )
        logger.info(f"Проверка фактов через {self.model_name}...")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096, "num_predict": 200}
        }
        
        try:
            r = requests.post(self.ollama_url, json=payload, timeout=60)
            r.raise_for_status()
            result = r.json().get("response", "").strip()
            
            if "ГАЛЛЮЦИНАЦИЯ" in result.upper() or "НАРУШЕНИЕ" in result.upper():
                logger.warning(f"Найдена ошибка: {result}")
                return False, result
            return True, ""
        except Exception as e:
            logger.error(f"Ошибка API Аудитора: {e}")
            return True, ""