FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

# Copy project
COPY . /app/

# Create necessary directories
RUN mkdir -p uploads models

# Expose port
EXPOSE 5000

# Run gunicorn
# 4 workers is a safe default for small-medium load
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
