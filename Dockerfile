FROM python:3.12-slim

# Не писать .pyc в контейнер и не буферизовать stdout (нужно для логов uvicorn)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Сначала только зависимости — используем кэш слоев Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем остальной код
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

# Запускаем API, а не интерактивный CLI (main.py ждет ввода и в контейнере падает)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
