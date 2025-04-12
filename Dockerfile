# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy everything to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
