
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8501
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install remaining dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application
COPY . .

EXPOSE 8501

CMD ["sh", "-c", "uvicorn app.api.main:app --host 127.0.0.1 --port 8000 & streamlit run app/frontend/streamlit.py --server.address 0.0.0.0 --server.port ${PORT}"]
