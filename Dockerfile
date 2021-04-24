FROM ubuntu:18.04
RUN apt-get update && apt-get upgrade -y && apt-get clean
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y
RUN apt-get -yqq install python3-pip
RUN pip3 install --upgrade setuptools
RUN pip3 install --upgrade pip
RUN pip3 install pytype
RUN pip3 install scikit-build
RUN pip3 install cmake
RUN pip3 install opencv-python==4.2.0.32
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
WORKDIR /home/ubuntu
ADD object_detection.py /home/ubuntu
ADD iWebLens_client.py /home/ubuntu
ADD reqfolders  /home/ubuntu
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
CMD ["python3" , "object_detection.py", "yolo_tiny_configs"]

