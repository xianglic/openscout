# Use an official Python runtime as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --upgrade flask werkzeug

COPY ./chatgpt_server.py /app/server.py

# Make port 5000 available to the world outside this container
EXPOSE 5000

ENTRYPOINT  ["python3", "/app/server.py"]