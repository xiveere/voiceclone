# Use the base SageMaker PyTorch image
FROM 885854791233.dkr.ecr.us-east-1.amazonaws.com/sagemaker-distribution-prod:gpu-1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN sudo apt-get update && sudo apt-get install -y \
    ffmpeg \
    espeak \
    && rm -rf /var/lib/apt/lists/*

# Make port 5000 available to the world outside this container
EXPOSE 8080

CMD gunicorn -w 4 -b 0.0.0.0:8080 generate_xtts:app