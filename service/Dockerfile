FROM python:3.10-slim

WORKDIR /app

 
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

 
COPY app.py .
 
COPY mlp_model_architecture.json .
COPY mlp_model.weights.h5 .
COPY mlp_model_scaler.pkl .

 
# COPY ../Projects ./Projects

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]