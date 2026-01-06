FROM python:3.12-slim

RUN apt-get update && apt-get -y upgrade && apt-get install -y --no-install-recommends \
    build-essential \
    # Indigo renderer runtime deps reported missing:
    libfreetype6 \
    libfontconfig1 \
    fonts-dejavu-core \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install Python deps
RUN pip install -r requirements.txt

# Make sure the dynamic loader can find Indigo’s bundled libs (libindigo.so, etc.)
# Path matches your environment: /usr/local/lib/python3.12/site-packages/indigo/lib/linux-x86_64
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/indigo/lib/linux-x86_64:$LD_LIBRARY_PATH


COPY . .

EXPOSE 8080

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8080", "app:app"]
