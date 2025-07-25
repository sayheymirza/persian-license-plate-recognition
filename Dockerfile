# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including OpenCV requirements
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for uploaded files and set permissions
RUN mkdir -p public && \
    chown -R www-data:www-data /app && \
    chmod -R 755 /app && \
    chmod 777 public

# Copy the application code (as root)
COPY --chown=www-data:www-data . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    UPLOAD_FOLDER=public \
    FILE_LIFETIME=60 \
    PORT=8080 \
    DEBUG=False

# Switch to non-root user
USER www-data

# Expose the port
EXPOSE 8080

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--threads", "2", "--timeout", "120", "app:app"]
