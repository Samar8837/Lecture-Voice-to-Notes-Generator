# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# ffmpeg is required for general audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code to the working directory
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variables for Streamlit to run correctly
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Run the app.py script when the container launches
CMD ["streamlit", "run", "app.py"]