# Use a specific, compatible Python version
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY ./backend/requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code
COPY ./backend /app

# Copy the model files
COPY ./model /app/model

# Expose the port the app runs on
EXPOSE 8000

# Define the command to run the application
# The host must be 0.0.0.0 to be accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
