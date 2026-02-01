FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn
RUN pip install --no-cache-dir pandas scikit-learn
RUN pip install --no-cache-dir tensorflow

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
