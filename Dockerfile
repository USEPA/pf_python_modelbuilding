FROM python:3.12-alpine

RUN apk update && apk upgrade && apk add --update alpine-sdk

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["waitress-serve", "--host", "0.0.0.0", "--port", "5000", "wsgi:app"]
