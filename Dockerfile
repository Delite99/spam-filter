# SpamGuard Production Dockerfile
FROM python:3.12-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=3030

WORKDIR /app

# Configure apt to retry on failure (helps with flaky Debian mirrors)
RUN echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80retries

# Install curl (healthcheck) and build-essential (for pip wheel compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU build first (avoids downloading the heavy CUDA variant)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install all remaining dependencies (torch is already satisfied above)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and assets
COPY src/ /app/src/
COPY models/ /app/models/
COPY reports/ /app/reports/
COPY data/ /app/data/

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 3030

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:3030/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "3030"]
