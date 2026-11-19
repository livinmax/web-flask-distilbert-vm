# Official Python image
FROM python:3.9-slim

# Application directory
WORKDIR /app

# Trained weight file
COPY trained_distilbert_pytorch.pt /app/

# Local tokenizer and config assets
COPY local_model_assets_multi /app/local_model_assets_multi

# Application files
COPY app.py /app/
COPY requirements.txt /app/
COPY templates /app/templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start the Flask app
CMD ["python", "app.py"]
