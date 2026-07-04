FROM python:3.10

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --default-timeout=300 --retries 20 torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --default-timeout=300 --retries 20 -r requirements.txt

COPY app.py .
COPY hausa.html .
COPY models ./models

EXPOSE 5003

ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=5003
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]