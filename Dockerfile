# 1. Base Image: Use an official lightweight Debian-based Python image
FROM python:3.11-slim

# 2. Set environment variables
# Prevent Python from writing .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Prevent Python from buffering stdout/stderr so logs appear immediately
ENV PYTHONUNBUFFERED=1

# 3. Install low-level system dependencies required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy dependencies first (Leverages Docker build cache)
COPY requirements.txt .

# 6. Install Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy application code and model artifacts into the container
COPY app/ ./app/
COPY artifacts/ ./artifacts/

# 8. Expose the port the API will listen on
EXPOSE 8000

# 9. Default command executed when container starts
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]