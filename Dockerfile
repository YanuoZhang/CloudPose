FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 60001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "60001"]