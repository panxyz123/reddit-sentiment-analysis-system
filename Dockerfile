# Use a slim Python image for a smaller footprint and better security
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from buffering stdout and stderr (better for Docker logging)
ENV PYTHONUNBUFFERED=1
# Set non-interactive mode for apt-get to avoid prompts during build
ENV DEBIAN_FRONTEND=noninteractive
# Ensure the /app directory is in the Python path for module resolution
ENV PYTHONPATH=/app

# Install system-level dependencies
# Added git/build-essential for potential C-extensions in ML libraries
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    bash \
    jq \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies separately to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire project into the container
# This includes the model directory, app.py, and utility scripts
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Start the application using Uvicorn
# --host 0.0.0.0 is required to make the service reachable from outside the container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]