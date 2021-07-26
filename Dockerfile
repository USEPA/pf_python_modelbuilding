# Import Docker image
FROM ubuntu:20.04

# Put source code in docker container
WORKDIR /app
COPY . /app

#Need python3.9 to run random forest using multicores...
RUN apt-get update
RUN apt-get install -y python3.9
RUN apt-get install -y python3-pip

# Need following to get scipy pip install to work:
RUN apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran

# Install pip
#RUN echo "Y" | apt install python3-pip

COPY requirements.txt .
# Install dependencies
#RUN /usr/bin/pip3 install -r ./requirements.txt
RUN pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run source code
CMD ["../usr/bin/python3", "./model_ws.py"]