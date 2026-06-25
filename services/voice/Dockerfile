FROM python:3.12-slim

WORKDIR /app

COPY services/voice/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY frontend /app/frontend
COPY libs /app/libs
COPY services /app/services

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.voice.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
