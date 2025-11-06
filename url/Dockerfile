FROM python:3.10-slim

WORKDIR /app

COPY app.py .
COPY templates ./templates
COPY static ./static

# Install all required dependencies
RUN pip install fastapi uvicorn jinja2 loguru python-multipart

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

