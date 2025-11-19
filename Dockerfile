FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
ENV TELEGRAM_TOKEN=""
ENV OPENAI_API_KEY=""
CMD ["python", "bot_mini_rag.py"]
