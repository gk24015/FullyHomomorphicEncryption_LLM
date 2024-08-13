# Use an official Python runtime as a parent image
FROM python:3.10.11-slim

# Set the working directory in the container
WORKDIR /encrypted_sentiment_analysis

RUN apt-get update && apt-get install -y \
    binutils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Run app.py when the container launches
CMD ["python", "app.py"]
