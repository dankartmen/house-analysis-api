# Базовый образ: лёгкий Python
FROM python:3.11-slim

# Установка системных зависимостей (если нужно, для pandas/sklearn)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и данные
COPY main.py .
COPY houses.csv .

# Запуск uvicorn (ASGI-сервер)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]