FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories and make them writable for HF Spaces (runs as uid 1000)
RUN mkdir -p /app/data/users /app/data/models && \
    chmod -R 777 /app/data

ENV PYTHONUNBUFFERED=1
ENV TF_NUM_INTEROP_THREADS=1
ENV TF_NUM_INTRAOP_THREADS=1
ENV OMP_NUM_THREADS=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "run.py"]
