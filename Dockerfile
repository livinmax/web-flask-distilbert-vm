# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл весов модели
COPY trained_distilbert_pytorch.pt /app/

# Копируем папку с локальными ассетами (конфиг, словарь и т.д.)
COPY local_model_assets_multi /app/local_model_assets_multi

# Копируем остальные файлы приложения
COPY app.py /app/
COPY requirements.txt /app/
COPY templates /app/templates/

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Указываем команду для запуска приложения
CMD ["python", "app.py"]
