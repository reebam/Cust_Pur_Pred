FROM python:3.9

WORKDIR /app

# Copy requirements from your subfolder to the docker build context
COPY tourism_project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app files
COPY . .

# Hugging Face Spaces run on port 7860
CMD ["streamlit", "run", "deployment/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
