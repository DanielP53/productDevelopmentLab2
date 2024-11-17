# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && apt-get clean

# Copy the application script and .env file to the container
COPY predict.py /app/
COPY .env /app/

# Create and install dependencies from requirements.txt
RUN echo "pandas\n" \
         "numpy\n" \
         "scikit-learn\n" \
         "xgboost\n" \
         "fastapi\n" \
         "uvicorn\n" \
         "python-dotenv\n" \
         "pyarrow\n" > requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

# Define the entry point for the container
ENTRYPOINT ["python", "predict.py"]
