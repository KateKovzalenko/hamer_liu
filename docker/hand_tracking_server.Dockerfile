# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /server

# --- THIS SECTION IS UPDATED ---
# Install a more complete set of system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
# -----------------------------

# Copy the server/requirements file first to leverage Docker cache
COPY server/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY server ./server

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV MEDIA_PIPE_DISABLE_GPU=1

# Run app.py when the container launches
CMD ["python", "-m", "server.server"]