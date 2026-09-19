FROM python:3.10-slim

WORKDIR /app

# Install essential C/C++ compilation tools for dependencies like scikit-learn / lightgbm
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code, scripts, and model artifacts
COPY . .

# Expose default Streamlit port
EXPOSE 8501

# Health check to ensure container status monitoring
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app bound to 0.0.0.0
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]