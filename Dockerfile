FROM --platform=linux/amd64 python:3.12-slim

RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    build-essential \
    libfreetype6 \
    libfontconfig1 \
    fonts-dejavu-core \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app:app"]
