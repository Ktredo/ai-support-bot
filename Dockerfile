# Используем легкий образ Python
FROM python:3.12-slim

# Устанавливаем системные зависимости (нужны для PyMuPDF)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала копируем только requirements, чтобы использовать кэш Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Запускаем бота
CMD ["python", "main.py"]
