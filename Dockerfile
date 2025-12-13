FROM python:3.12-slim

RUN apt-get -y update && apt-get -y upgrade && apt-get -y install build-essential

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "model_ws:app"]
