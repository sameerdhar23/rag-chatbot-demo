FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps sometimes needed for tokenizers / torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code + artifacts into image
COPY app /app/app
COPY artifacts /app/artifacts

# Render sets PORT; gunicorn binds to it
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-8080} app.main:app"]

