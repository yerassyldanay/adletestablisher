# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Set the time zone to Almaty, Kazakhstan
ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Copy the code into the container
COPY . /app

# Set the working directory
WORKDIR /app

RUN ls -la

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Set the command to run the code
CMD [ "python", "main.py" ]
