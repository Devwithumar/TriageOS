FROM python:3.12-slim

WORKDIR /app

COPY services/conversation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY libs /app/libs
COPY services /app/services

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.conversation.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
