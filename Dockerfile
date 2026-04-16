# --- Base Image ---
FROM python:3.10-slim AS base

# Install System Dependencies (OCR, PDF rendering, and OpenGL libs)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download BOTH NLTK tokenizer packages at build time.
# - 'punkt'     : still needed at runtime by sent_tokenize() for english.pickle
# - 'punkt_tab' : needed by NLTK 3.8+ internal format validation
# Both are required; removing either one causes a runtime error.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# --- Backend Application ---
FROM base AS backend
COPY app /app/app
RUN mkdir -p data/uploaded_pdfs data/vector_index data/cache
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Frontend Application ---
FROM base AS frontend
COPY frontend /app/frontend
EXPOSE 8501
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
